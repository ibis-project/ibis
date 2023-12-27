from __future__ import annotations

from ibis.formats.pandas import PandasData


class PostgresPandasData(PandasData):
    @classmethod
    def convert_GeoSpatial(cls, s, dtype, pandas_type):
        import geopandas as gpd
        import shapely as shp

        return gpd.GeoSeries(shp.from_wkb(s.map(bytes, na_action="ignore")))

    convert_Point = (
        convert_LineString
    ) = (
        convert_Polygon
    ) = (
        convert_MultiLineString
    ) = convert_MultiPoint = convert_MultiPolygon = convert_GeoSpatial

    @classmethod
    def convert_Binary(cls, s, dtype, pandas_type):
        return s.map(bytes, na_action="ignore")
