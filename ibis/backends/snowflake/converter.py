from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

import pyarrow as pa

from ibis.formats.pandas import PandasData
from ibis.formats.pyarrow import PyArrowData

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    from ibis.expr.schema import Schema


class JSONScalar(pa.ExtensionScalar):
    def as_py(self):
        value = self.value
        if value is None:
            return value
        else:
            return json.loads(value.as_py())


class JSONArray(pa.ExtensionArray):
    pass


class JSONType(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.string(), "ibis.json")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

    def __arrow_ext_class__(self):
        return JSONArray

    def __arrow_ext_scalar_class__(self):
        return JSONScalar


PYARROW_JSON_TYPE = JSONType()
pa.register_extension_type(PYARROW_JSON_TYPE)


class SnowflakePandasData(PandasData):
    @classmethod
    def convert_Timestamp_element(cls, dtype):
        return datetime.datetime.fromisoformat

    @classmethod
    def convert_Date_element(cls, dtype):
        return datetime.date.fromisoformat

    @classmethod
    def convert_Time_element(cls, dtype):
        return datetime.time.fromisoformat

    @classmethod
    def convert_JSON(cls, s, dtype, pandas_type):
        converter = cls.convert_JSON_element(dtype)
        return s.map(converter, na_action="ignore").astype("object")

    @classmethod
    def convert_Array(cls, s, dtype, pandas_type):
        raw_json_objects = cls.convert_JSON(s, dtype, pandas_type)
        return super().convert_Array(raw_json_objects, dtype, pandas_type)

    @classmethod
    def convert_Map(cls, s, dtype, pandas_type):
        raw_json_objects = cls.convert_JSON(s, dtype, pandas_type)
        return super().convert_Map(raw_json_objects, dtype, pandas_type)

    @classmethod
    def convert_Struct(cls, s, dtype, pandas_type):
        raw_json_objects = cls.convert_JSON(s, dtype, pandas_type)
        return super().convert_Struct(raw_json_objects, dtype, pandas_type)


class SnowflakePyArrowData(PyArrowData):
    @classmethod
    def convert_table(cls, table: pa.Table, schema: Schema) -> pa.Table:
        columns = [cls.convert_column(table[name], typ) for name, typ in schema.items()]
        return pa.Table.from_arrays(columns, names=schema.names)

    @classmethod
    def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
        if dtype.is_json() or dtype.is_array() or dtype.is_map() or dtype.is_struct():
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()

            return pa.ExtensionArray.from_storage(PYARROW_JSON_TYPE, column)
        return super().convert_column(column, dtype)
