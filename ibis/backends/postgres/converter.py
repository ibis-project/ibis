from __future__ import annotations

import json

from ibis.formats.pandas import PandasData


class PostgresPandasData(PandasData):
    @classmethod
    def convert_GeoSpatial(cls, s, dtype, pandas_type):
        import geopandas as gpd
        import shapely as shp

        return gpd.GeoSeries(shp.from_wkb(s.map(bytes, na_action="ignore")))

    convert_Point = convert_LineString = convert_Polygon = convert_MultiLineString = (
        convert_MultiPoint
    ) = convert_MultiPolygon = convert_GeoSpatial

    @classmethod
    def convert_Binary(cls, s, dtype, pandas_type):
        return s.map(bytes, na_action="ignore")

    @classmethod
    def convert_Map_element(cls, dtype):
        convert_key = cls.get_element_converter(dtype.key_type)
        convert_value = cls.get_element_converter(dtype.value_type)

        def convert(raw_row):
            if raw_row is None:
                return raw_row

            row = json.loads(raw_row)
            return dict(
                zip(map(convert_key, row.keys()), map(convert_value, row.values()))
            )

        return convert
