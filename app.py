"""Streamlit UI for the DMRetriever spatiotemporal search pipeline."""

from datetime import datetime, time, timedelta, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline.external_maps import EXTERNAL_MAPS
from pipeline.pipeline import SearchRequest, run_search
from pipeline.ranker import DMRetrieverRanker
from pipeline.sources import gdelt

st.set_page_config(
    page_title="DMRetriever Spatiotemporal Search",
    layout="wide",
)
st.title("DMRetriever Spatiotemporal Search")
st.caption(
    "Query GDELT news within a space + time window, optionally re-ranked by DMRetriever."
)

page_search, page_maps = st.tabs(["🔎 Search", "🗺️ Reference flood maps"])


@st.cache_resource(show_spinner="Loading DMRetriever model…")
def load_ranker(model_id: str) -> DMRetrieverRanker:
    return DMRetrieverRanker(model_id=model_id)


with st.sidebar:
    st.subheader("Query")
    query = st.text_input("Search query", value="flood evacuation")

    st.subheader("Time window")
    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=7)
    date_range = st.date_input(
        "Date range (UTC)",
        value=(default_start, today),
        max_value=today,
    )

    st.subheader("Filters")
    sourcecountry = st.text_input(
        "Source country (FIPS code, e.g. US, JA, UK)", value=""
    )
    sourcelang = st.text_input("Source language (eng, fre, spa, …)", value="")
    theme = st.text_input(
        "GDELT theme (e.g. NATURAL_DISASTER, FLOOD)", value=""
    )
    max_records = st.slider("Max records", 10, 250, 75, step=5)

    st.subheader("Re-ranking")
    use_ranker = st.checkbox("Re-rank with DMRetriever", value=True)
    model_id = st.selectbox(
        "DMRetriever checkpoint",
        [
            "DMIR01/DMRetriever-33M",
            "DMIR01/DMRetriever-109M",
            "DMIR01/DMRetriever-335M",
        ],
        index=0,
        disabled=not use_ranker,
        help="Lightweight 33M is the fastest. Larger checkpoints score better but need more RAM/VRAM.",
    )
    top_k = st.slider(
        "Top K after re-rank", 5, 100, 25, disabled=not use_ranker
    )

    run = st.button("Search", type="primary", use_container_width=True)


def _date_range_to_window(value):
    if isinstance(value, tuple) and len(value) == 2:
        start_d, end_d = value
    elif isinstance(value, tuple) and len(value) == 1:
        start_d = end_d = value[0]
    else:
        start_d = end_d = value
    start_dt = datetime.combine(start_d, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_d, time.max).replace(tzinfo=timezone.utc)
    return start_dt, end_dt


with page_search:
    if run:
        if not query.strip() and not sourcecountry.strip() and not theme.strip():
            st.warning("Enter a query or at least one filter.")
            st.stop()

        start_dt, end_dt = _date_range_to_window(date_range)

        req = SearchRequest(
            query=query.strip(),
            start=start_dt,
            end=end_dt,
            sourcecountry=sourcecountry.strip() or None,
            sourcelang=sourcelang.strip() or None,
            theme=theme.strip() or None,
            max_records=max_records,
        )

        ranker = load_ranker(model_id) if use_ranker else None

        with st.spinner("Fetching from GDELT…"):
            try:
                results = run_search(
                    req,
                    ranker=ranker,
                    top_k=top_k if use_ranker else None,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        st.success(f"Retrieved {len(results)} article(s).")

        tab_table, tab_map, tab_raw = st.tabs(["Results", "Map", "Raw JSON"])

        with tab_table:
            if not results:
                st.info("No results.")
            else:
                df = pd.DataFrame(
                    [
                        {
                            "score": round(r.score, 4) if r.score is not None else None,
                            "published_at": r.published_at,
                            "title": r.title,
                            "url": r.url,
                            "domain": r.domain,
                            "country": r.country,
                            "language": r.language,
                        }
                        for r in results
                    ]
                )
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "url": st.column_config.LinkColumn("url"),
                        "published_at": st.column_config.DatetimeColumn(
                            "published_at", format="YYYY-MM-DD HH:mm Z"
                        ),
                    },
                )
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode(),
                    "gdelt_results.csv",
                    "text/csv",
                )

        with tab_map:
            st.caption(
                "Geographic mentions for the same query, via the GDELT GEO 2.0 API."
            )
            try:
                geo = gdelt.geo_lookup(
                    req.query,
                    start=start_dt,
                    end=end_dt,
                    sourcecountry=req.sourcecountry,
                    sourcelang=req.sourcelang,
                    theme=req.theme,
                )
            except Exception as e:
                st.warning(f"GEO API failed: {e}")
                geo = {"features": []}
            points = []
            for f in geo.get("features", []):
                geom = f.get("geometry", {}) or {}
                if geom.get("type") != "Point":
                    continue
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    continue
                lon, lat = coords[0], coords[1]
                props = f.get("properties", {}) or {}
                points.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "name": props.get("name", ""),
                        "count": props.get("count", 1),
                    }
                )
            if not points:
                st.info("No geocoded mentions for this query/window.")
            else:
                df_map = pd.DataFrame(points)
                st.map(df_map[["lat", "lon"]])
                with st.expander(f"Locations ({len(df_map)})"):
                    st.dataframe(df_map, use_container_width=True, hide_index=True)

        with tab_raw:
            st.json([r.raw for r in results[:50]])
    else:
        st.info("Set query parameters in the sidebar and click **Search**.")


with page_maps:
    st.caption(
        "External flood-monitoring dashboards, embedded as iframes. "
        "Some sites (Google Flood Hub especially) set X-Frame-Options that block "
        "iframing — if a panel renders blank, use the **Open in new tab** link."
    )

    map_tabs = st.tabs([m.name for m in EXTERNAL_MAPS])
    for tab, m in zip(map_tabs, EXTERNAL_MAPS):
        with tab:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{m.name}**  \n{m.description}")
            with cols[1]:
                st.link_button("↗ Open in new tab", m.url, use_container_width=True)
            components.iframe(m.url, height=m.height, scrolling=True)
