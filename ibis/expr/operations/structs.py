from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from public import public

from ibis.common.validators import immutable_property
from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Value, distinct_roots

if TYPE_CHECKING:
    import ibis.expr.operations as ops


@public
class StructField(Value):
    arg = rlz.struct
    field = rlz.instance_of(str)

    output_shape = rlz.shape_like("arg")

    @immutable_property
    def output_dtype(self) -> dt.DataType:
        struct_dtype = self.arg.type()
        value_dtype = struct_dtype[self.field]
        return value_dtype

    def resolve_name(self) -> str:
        return self.field

    def has_resolved_name(self) -> bool:
        return True


@public
class StructColumn(Value):
    names = rlz.tuple_of(rlz.instance_of(str), min_length=1)
    values = rlz.tuple_of(rlz.any, min_length=1)

    output_shape = rlz.Shape.COLUMNAR

    @immutable_property
    def output_dtype(self) -> dt.DataType:
        return dt.Struct.from_tuples(
            zip(self.names, (value.type() for value in self.values))
        )

    def root_tables(self) -> Sequence[ops.TableNode]:
        return distinct_roots(*self.values)
