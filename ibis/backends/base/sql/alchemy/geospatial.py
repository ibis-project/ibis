from __future__ import annotations

from importlib.util import find_spec as _find_spec

geospatial_supported = (
    _find_spec("geoalchemy2") is not None and _find_spec("geopandas") is not None
)
__all__ = ["geospatial_supported"]
