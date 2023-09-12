from __future__ import annotations

from ibis.formats.pandas import PandasData


class SnowflakePandasData(PandasData):
    convert_Struct = staticmethod(PandasData.convert_JSON)
    convert_Array = staticmethod(PandasData.convert_JSON)
    convert_Map = staticmethod(PandasData.convert_JSON)
