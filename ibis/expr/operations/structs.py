from __future__ import annotations

from public import public

from ibis.common.validators import immutable_property
from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Value


@public
class StructField(Value):
    arg = rlz.struct
    field = rlz.instance_of(str)

    output_shape = rlz.shape_like("arg")

    @immutable_property
    def output_dtype(self) -> dt.DataType:
        struct_dtype = self.arg.output_dtype
        value_dtype = struct_dtype[self.field]
        return value_dtype

    @property
    def name(self) -> str:
        return self.field


@public
class StructColumn(Value):
    names = rlz.tuple_of(rlz.instance_of(str), min_length=1)
    values = rlz.nodes_of(rlz.any, min_length=1)

    output_shape = rlz.Shape.COLUMNAR

    @immutable_property
    def output_dtype(self) -> dt.DataType:
        dtypes = (value.output_dtype for value in self.values)
        return dt.Struct.from_tuples(zip(self.names, dtypes))
