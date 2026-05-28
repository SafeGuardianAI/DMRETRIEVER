from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class SourceResult:
    source: str
    title: str
    url: str
    snippet: str = ""
    published_at: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    country: Optional[str] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    score: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
