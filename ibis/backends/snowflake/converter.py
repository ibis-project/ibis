from ibis.formats.pandas import PandasConverter


class SnowflakePandasConverter(PandasConverter):
    convert_Struct = staticmethod(PandasConverter.convert_JSON)
    convert_Array = staticmethod(PandasConverter.convert_JSON)
    convert_Map = staticmethod(PandasConverter.convert_JSON)
