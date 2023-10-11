from __future__ import annotations

from ibis.formats.pandas import PandasData


class SnowflakePandasData(PandasData):
    @staticmethod
    def convert_JSON(s, dtype, pandas_type):
        converter = SnowflakePandasData.convert_JSON_element(dtype)
        return s.map(converter, na_action="ignore").astype("object")

    convert_Struct = convert_Array = convert_Map = convert_JSON
