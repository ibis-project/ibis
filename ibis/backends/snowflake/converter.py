from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    from ibis.expr.schema import Schema


class JSONScalar(pa.ExtensionScalar):
    def as_py(self, **_):
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


try:
    from ibis.formats.pandas import PandasData
except ModuleNotFoundError:
    pass
else:

    class SnowflakePandasData(PandasData):
        @classmethod
        def convert_Timestamp_element(cls, dtype):
            return pd.Timestamp.fromisoformat

        @classmethod
        def convert_Date_element(cls, dtype):
            return pd.Timestamp.fromisoformat

        @classmethod
        def convert_Time_element(cls, dtype):
            return pd.Timestamp.fromisoformat

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


try:
    from ibis.formats.pyarrow import PyArrowData
except ModuleNotFoundError:
    pass
else:

    class SnowflakePyArrowData(PyArrowData):
        @classmethod
        def convert_table(cls, table: pa.Table, schema: Schema) -> pa.Table:
            columns = [
                cls.convert_column(table[name], typ) for name, typ in schema.items()
            ]
            return pa.Table.from_arrays(
                columns,
                schema=pa.schema(
                    [
                        pa.field(name, array.type, nullable=ibis_dtype.nullable)
                        for array, (name, ibis_dtype) in zip(columns, schema.items())
                    ]
                ),
            )

        @classmethod
        def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
            # Fixed-length numeric arrays (e.g. Snowflake VECTOR(FLOAT, N))
            # deserialize natively to pyarrow ``fixed_size_list<element>[N]``
            # via the Snowflake Python connector. They are not JSON-encoded
            # and so should bypass the JSON-extension wrapping below; without
            # this guard, the JSON-extension path tries to cast the
            # ``fixed_size_list`` column to ``utf8`` and raises
            # ``ArrowNotImplementedError: Unsupported cast from
            # fixed_size_list<...> to utf8``.
            if dtype.is_array() and dtype.length is not None:
                return column
            if (
                dtype.is_json()
                or dtype.is_array()
                or dtype.is_map()
                or dtype.is_struct()
            ):
                if isinstance(column, pa.ChunkedArray):
                    column = column.combine_chunks()

                return pa.ExtensionArray.from_storage(PYARROW_JSON_TYPE, column)
            return super().convert_column(column, dtype)

        @classmethod
        def convert_scalar(cls, scalar: pa.Scalar, dtype: dt.DataType) -> pa.Scalar:
            # Same fixed-size-list pass-through as ``convert_column``: VECTOR
            # scalars are already native fixed-length pyarrow lists and
            # don't need the JSON-string extension wrap.
            if dtype.is_array() and dtype.length is not None:
                return scalar
            if (
                dtype.is_json()
                or dtype.is_array()
                or dtype.is_map()
                or dtype.is_struct()
            ):
                return pa.ExtensionScalar.from_storage(PYARROW_JSON_TYPE, scalar)
            return super().convert_scalar(scalar, dtype)
