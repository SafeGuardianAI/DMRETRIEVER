"""Spatiotemporal search orchestrator.

Today this fans out to GDELT only. The SearchRequest schema and run_search
seam are designed so additional sources (X, YouTube, CrossRef, …) can be
added behind the same interface without touching the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from .sources import gdelt
from .sources.base import SourceResult

if TYPE_CHECKING:
    from .ranker import DMRetrieverRanker


@dataclass
class SearchRequest:
    query: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    sourcecountry: Optional[str] = None
    sourcelang: Optional[str] = None
    theme: Optional[str] = None
    max_records: int = 75


def run_search(
    req: SearchRequest,
    ranker: Optional["DMRetrieverRanker"] = None,
    top_k: Optional[int] = None,
) -> List[SourceResult]:
    results = gdelt.search_doc(
        query=req.query,
        start=req.start,
        end=req.end,
        sourcecountry=req.sourcecountry,
        sourcelang=req.sourcelang,
        theme=req.theme,
        max_records=req.max_records,
    )
    if ranker is not None and req.query.strip():
        results = ranker.rerank(req.query, results, top_k=top_k)
    elif top_k is not None:
        results = results[:top_k]
    return results
