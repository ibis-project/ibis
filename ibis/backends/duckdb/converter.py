from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


try:
    from ibis.formats.pandas import PandasData
except ModuleNotFoundError:
    pass
else:

    class DuckDBPandasData(PandasData):
        @staticmethod
        def convert_Array(s, dtype, pandas_type):
            return s.replace(float("nan"), None)


try:
    from ibis.formats.pyarrow import PyArrowData
except ModuleNotFoundError:
    pass
else:

    class DuckDBPyArrowData(PyArrowData):
        @classmethod
        def convert_scalar(cls, scalar: pa.Scalar, dtype: dt.DataType) -> pa.Scalar:
            if dtype.is_null():
                return pa.scalar(None)
            return super().convert_scalar(scalar, dtype)

        @classmethod
        def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
            if dtype.is_null():
                return pa.nulls(len(column))
            return super().convert_column(column, dtype)
