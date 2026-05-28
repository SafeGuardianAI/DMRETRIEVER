"""Spatiotemporal search pipeline.

Submodules are imported lazily so `pipeline.sources` works in
environments without torch / transformers installed.
"""

from .pipeline import SearchRequest, run_search

__all__ = ["SearchRequest", "run_search"]
