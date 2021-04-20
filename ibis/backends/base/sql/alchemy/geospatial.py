try:
    import geoalchemy2  # noqa F401
    import geoalchemy2.shape  # noqa F401
    import geopandas  # noqa F401
except ImportError:
    geospatial_supported = False
else:
    geospatial_supported = True
