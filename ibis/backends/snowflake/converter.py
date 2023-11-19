from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from ibis.formats.pandas import PandasData
from ibis.formats.pyarrow import PYARROW_JSON_TYPE, PyArrowData

if TYPE_CHECKING:
    import pyarrow as pa

    import ibis.expr.datatypes as dt
    from ibis.expr.schema import Schema


class SnowflakePandasData(PandasData):
    @staticmethod
    def convert_JSON(s, dtype, pandas_type):
        converter = SnowflakePandasData.convert_JSON_element(dtype)
        return s.map(converter, na_action="ignore").astype("object")

    convert_Struct = convert_Map = convert_JSON

    @staticmethod
    def get_element_converter(dtype):
        funcgen = getattr(
            SnowflakePandasData,
            f"convert_{type(dtype).__name__}_element",
            lambda _: lambda x: x,
        )
        return funcgen(dtype)

    def convert_Timestamp_element(dtype):
        return lambda values: list(map(datetime.datetime.fromisoformat, values))

    def convert_Date_element(dtype):
        return lambda values: list(map(datetime.date.fromisoformat, values))

    def convert_Time_element(dtype):
        return lambda values: list(map(datetime.time.fromisoformat, values))

    @staticmethod
    def convert_Array(s, dtype, pandas_type):
        raw_json_objects = SnowflakePandasData.convert_JSON(s, dtype, pandas_type)
        converter = SnowflakePandasData.get_element_converter(dtype.value_type)
        return raw_json_objects.map(converter, na_action="ignore")


class SnowflakePyArrowData(PyArrowData):
    @classmethod
    def convert_table(cls, table: pa.Table, schema: Schema) -> pa.Table:
        import pyarrow as pa

        columns = [cls.convert_column(table[name], typ) for name, typ in schema.items()]
        return pa.Table.from_arrays(columns, names=schema.names)

    @classmethod
    def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
        if dtype.is_json() or dtype.is_array() or dtype.is_map() or dtype.is_struct():
            import pyarrow as pa

            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()

            return pa.ExtensionArray.from_storage(PYARROW_JSON_TYPE, column)
        return super().convert_column(column, dtype)
