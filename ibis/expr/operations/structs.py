from __future__ import annotations

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Value


@public
class StructField(Value):
    arg: Value[dt.Struct]
    field: str

    output_shape = rlz.shape_like("arg")

    @attribute.default
    def output_dtype(self) -> dt.DataType:
        struct_dtype = self.arg.output_dtype
        value_dtype = struct_dtype[self.field]
        return value_dtype

    @property
    def name(self) -> str:
        return self.field


@public
class StructColumn(Value):
    names: VarTuple[str]
    values: VarTuple[Value]

    output_shape = ds.columnar

    @attribute.default
    def output_dtype(self) -> dt.DataType:
        dtypes = (value.output_dtype for value in self.values)
        return dt.Struct.from_tuples(zip(self.names, dtypes))
