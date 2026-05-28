# Spatiotemporal Search Pipeline

Disaster-domain news search over a space + time window, with optional
DMRetriever re-ranking. Ships with a Streamlit UI (`app.py`) and a
small Python API.

## What's in scope today

- **GDELT 2.0 DOC API** — article search with publication time,
  source country, source language, and GKG theme filters.
- **GDELT 2.0 GEO API** — geocoded mentions for the same query,
  rendered as map points.
- **DMRetriever re-ranking** — cosine similarity between the user
  query and each retrieved title, computed from DMRetriever
  mean-pooled embeddings.

GDELT requires no API key. The `SearchRequest` / `run_search` seam is
designed so X, YouTube, and academic sources can be plugged in behind
the same interface without UI changes.

## Install

```bash
pip install -r requirements-pipeline.txt
```

## Run the UI

```bash
streamlit run app.py
```

### Startup noise about `torchvision`

Streamlit's file watcher walks every imported module and probes its
`__path__`. The `transformers` package lazy-imports vision submodules
(vitpose, yolos, zoedepth, …) on attribute access, several of which
need `torchvision`. Since DMRetriever uses only text encoders we don't
install torchvision, so those probes raise `ModuleNotFoundError`.

The app still runs; the tracebacks are watcher noise, not crashes.
`.streamlit/config.toml` disables the watcher (`fileWatcherType = "none"`)
to silence them. If you're actively editing `app.py` and want hot
reload back, either remove that line or run:

```bash
pip install torchvision
```

## Programmatic use

```python
from datetime import datetime, timedelta, timezone

from pipeline import SearchRequest, run_search
from pipeline.ranker import DMRetrieverRanker

end = datetime.now(timezone.utc)
start = end - timedelta(days=7)

req = SearchRequest(
    query="flood evacuation",
    start=start,
    end=end,
    sourcecountry="US",       # FIPS country code
    sourcelang="eng",
    theme="NATURAL_DISASTER", # GDELT GKG theme
    max_records=100,
)

ranker = DMRetrieverRanker("DMIR01/DMRetriever-33M")
results = run_search(req, ranker=ranker, top_k=25)

for r in results[:5]:
    print(f"{r.score:.3f}  {r.published_at}  {r.title}\n  {r.url}")
```

## Secrets

`pipeline.config.get_secret(name)` reads `st.secrets` first, then the
process environment (which `python-dotenv` populates from a top-level
`.env`). Copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in keys for any future sources you
add.

## Adding another source

1. Create `pipeline/sources/<name>.py` with a function that returns
   `List[SourceResult]`.
2. Wire it into `pipeline.pipeline.run_search` and extend
   `SearchRequest` with the new filter fields.
3. Add UI controls in `app.py` and let the existing ranker / table /
   map handle the rest.
