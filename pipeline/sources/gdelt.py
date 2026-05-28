"""GDELT 2.0 DOC + GEO API client.

DOC API:  free-text article search with time-window and per-article metadata
          (publication date, source country, language, domain).
GEO API:  geocoded mentions for the same query, returned as GeoJSON points
          suitable for mapping.

Both APIs are free and require no key.
Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
      https://blog.gdeltproject.org/announcing-the-geo-2-0-api/
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .base import SourceResult

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"

_USER_AGENT = "DMRetriever-spatiotemporal/0.1 (+https://github.com/safeguardianai/dmretriever)"


def _fmt_dt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _parse_seendate(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _build_query(
    query: str,
    sourcecountry: Optional[str],
    sourcelang: Optional[str],
    theme: Optional[str],
) -> str:
    parts: List[str] = []
    q = (query or "").strip()
    if q:
        parts.append(q)
    if sourcecountry:
        parts.append(f"sourcecountry:{sourcecountry.strip()}")
    if sourcelang:
        parts.append(f"sourcelang:{sourcelang.strip()}")
    if theme:
        parts.append(f"theme:{theme.strip()}")
    return " ".join(parts).strip()


def search_doc(
    query: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    sourcecountry: Optional[str] = None,
    sourcelang: Optional[str] = None,
    theme: Optional[str] = None,
    max_records: int = 75,
    sort: str = "datedesc",
    timeout: int = 30,
) -> List[SourceResult]:
    """Query the GDELT DOC 2.0 API and return normalized SourceResults."""
    q = _build_query(query, sourcecountry, sourcelang, theme)
    if not q:
        raise ValueError("GDELT requires at least one query term or filter.")

    params = {
        "query": q,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max(1, min(250, int(max_records)))),
        "sort": sort,
    }
    if start:
        params["startdatetime"] = _fmt_dt(start)
    if end:
        params["enddatetime"] = _fmt_dt(end)

    resp = requests.get(
        DOC_API,
        params=params,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()

    try:
        data = resp.json() if resp.text.strip() else {}
    except ValueError as e:
        raise RuntimeError(f"GDELT returned non-JSON response: {resp.text[:200]!r}") from e

    out: List[SourceResult] = []
    for art in data.get("articles", []):
        out.append(
            SourceResult(
                source="gdelt",
                title=art.get("title", "") or "",
                url=art.get("url", "") or "",
                snippet=art.get("title", "") or "",
                published_at=_parse_seendate(art.get("seendate")),
                country=art.get("sourcecountry"),
                language=art.get("language"),
                domain=art.get("domain"),
                raw=art,
            )
        )
    return out


def geo_lookup(
    query: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    sourcecountry: Optional[str] = None,
    sourcelang: Optional[str] = None,
    theme: Optional[str] = None,
    mode: str = "PointData",
    timeout: int = 30,
) -> Dict[str, Any]:
    """Fetch geocoded mentions as GeoJSON for the same query."""
    q = _build_query(query, sourcecountry, sourcelang, theme)
    if not q:
        return {"type": "FeatureCollection", "features": []}

    params = {
        "query": q,
        "mode": mode,
        "format": "GeoJSON",
    }
    if start:
        params["startdatetime"] = _fmt_dt(start)
    if end:
        params["enddatetime"] = _fmt_dt(end)

    resp = requests.get(
        GEO_API,
        params=params,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    if not resp.text.strip():
        return {"type": "FeatureCollection", "features": []}
    try:
        return resp.json()
    except ValueError:
        return {"type": "FeatureCollection", "features": []}
