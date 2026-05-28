"""Reference flood-monitoring map embeds.

Each entry is rendered as an iframe in the Streamlit UI. Many of these
sites set X-Frame-Options or CSP frame-ancestors that block iframing
(Google services in particular almost always do), so every panel also
exposes the raw URL as an "open in new tab" fallback.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ExternalMap:
    name: str
    url: str
    description: str
    height: int = 650


COPERNICUS_GLOFAS_URL = (
    "https://global-flood.emergency.copernicus.eu/react/map?%7B%22c%22%3A%2210051419"
    ".90608917%2C2811632.5912748193%22%2C%22z%22%3A%2210%22%2C%22layers%22%3A%22%257B"
    "%2522id%2522%253A%2522https%253A%252F%252Fglobal-flood.emergency.copernicus.eu"
    "%252Fproxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS%2524Flood%2520Risk%2524"
    "RapidImpactAssessment%2522%252C%2522visible%2522%253Atrue%252C%2522opacity%2522"
    "%253A0.8%257D%2C%257B%2522id%2522%253A%2522https%253A%252F%252Fglobal-flood"
    ".emergency.copernicus.eu%252Fproxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS"
    "%2524Flood%2520Risk%2524RapidFloodMapping%2522%252C%2522visible%2522%253Atrue"
    "%252C%2522opacity%2522%253A0.8%257D%2C%257B%2522id%2522%253A%2522https%253A%252F"
    "%252Fglobal-flood.emergency.copernicus.eu%252Fproxy%252F%253Fsrv%253Dows%2526"
    "SERVICE%253DWMS%2524Flood%2520Risk%2524critinfra_assets%2522%252C%2522visible"
    "%2522%253Atrue%252C%2522opacity%2522%253A0.8%257D%2C%257B%2522id%2522%253A%2522"
    "https%253A%252F%252Fglobal-flood.emergency.copernicus.eu%252Fproxy%252F%253Fsrv"
    "%253Dows%2526SERVICE%253DWMS%2524Hydrological%2524FloodSummary1_30%2522%252C"
    "%2522visible%2522%253Atrue%252C%2522opacity%2522%253A0.8%257D%2C%257B%2522id"
    "%2522%253A%2522https%253A%252F%252Fglobal-flood.emergency.copernicus.eu%252F"
    "proxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS%2524Hydrological%2524sumAL42EGE"
    "%2522%252C%2522visible%2522%253Atrue%252C%2522opacity%2522%253A0.8%257D%2C%257B"
    "%2522id%2522%253A%2522https%253A%252F%252Fglobal-flood.emergency.copernicus.eu"
    "%252Fproxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS%2524Hydrological%2524"
    "sumAL43EGE%2522%252C%2522visible%2522%253Atrue%252C%2522opacity%2522%253A0.8"
    "%257D%2C%257B%2522id%2522%253A%2522https%253A%252F%252Fglobal-flood.emergency"
    ".copernicus.eu%252Fproxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS%2524Static"
    "%2524MajorRivers1%2522%252C%2522visible%2522%253Atrue%252C%2522opacity%2522"
    "%253A0.8%257D%2C%257B%2522id%2522%253A%2522https%253A%252F%252Fglobal-flood"
    ".emergency.copernicus.eu%252Fproxy%252F%253Fsrv%253Dows%2526SERVICE%253DWMS"
    "%2524Static%2524MajorRiverBasins%2522%252C%2522visible%2522%253Atrue%252C"
    "%2522opacity%2522%253A0.8%257D%2C%257B%2522id%2522%253A%2522https%253A%252F"
    "%252Fglobal-flood.emergency.copernicus.eu%252Fproxy%252F%253Fsrv%253Dows%2526"
    "SERVICE%253DWMS%2524Meteorological%2524RainAnimationGLOFAS%2522%252C%2522"
    "visible%2522%253Atrue%252C%2522opacity%2522%253A0.8%257D%2C%257B%2522id%2522"
    "%253A%2522https%253A%252F%252Fglobal-flood.emergency.copernicus.eu%252Fproxy"
    "%252F%253Fsrv%253Dows%2526SERVICE%253DWMS%2524Static%2524WorldBoundaries"
    "%2522%252C%2522visible%2522%253Atrue%252C%2522opacity%2522%253A0.8%257D%22"
    "%2C%22lview%22%3A%22GROUP%22%2C%22base_map%22%3A%221%22%7D"
)

GOOGLE_FLOOD_HUB_URL = (
    "https://sites.research.google/floods/l/24.375907911184946/92.7411393245864/"
    "9.315592851638794/g/BWDB_SW270?layers=3&hide_layers=2"
)

FFWC_BD_URL = "https://ffwc.gov.bd/app/home"


EXTERNAL_MAPS: List[ExternalMap] = [
    ExternalMap(
        name="Copernicus EMS — Global Flood Awareness (GloFAS)",
        url=COPERNICUS_GLOFAS_URL,
        description=(
            "Pre-configured layers: Rapid Impact Assessment, Rapid Flood Mapping, "
            "critical infrastructure, 1–30d flood summary, GloFAS rain animation, "
            "major rivers & basins."
        ),
    ),
    ExternalMap(
        name="Google Flood Hub — Bangladesh (BWDB_SW270)",
        url=GOOGLE_FLOOD_HUB_URL,
        description=(
            "Google Research flood forecast view centered on north-east Bangladesh. "
            "Note: Google typically blocks iframing — use the external link."
        ),
    ),
    ExternalMap(
        name="FFWC — Bangladesh Flood Forecasting & Warning Centre",
        url=FFWC_BD_URL,
        description="Official BD government flood forecasts and station readings.",
    ),
]
