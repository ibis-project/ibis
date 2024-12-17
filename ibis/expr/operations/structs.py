"""Operations for working with structs."""

from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import ValidationError, attribute
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Value


@public
class StructField(Value):
    """Extract a field from a struct value."""

    arg: Value[dt.Struct]
    field: str

    shape = rlz.shape_like("arg")

    @attribute
    def dtype(self) -> dt.DataType:
        struct_dtype = self.arg.dtype
        value_dtype = struct_dtype[self.field]
        return value_dtype

    @property
    def name(self) -> str:
        return self.field


@public
class StructColumn(Value):
    """Construct a struct column from literals or expressions."""

    names: VarTuple[str]
    values: VarTuple[Value]

    shape = rlz.shape_like("values")

    def __init__(self, names, values):
        if len(names) != len(values):
            raise ValidationError(
                f"Length of names ({len(names)}) does not match length of "
                f"values ({len(values)})"
            )
        super().__init__(names=names, values=values)

    @property
    def name(self) -> str:
        pairs = ", ".join(
            f"{name!r}: {op.name}" for name, op in zip(self.names, self.values)
        )
        return f"{self.__class__.__name__}({{{pairs}}})"

    @attribute
    def dtype(self) -> dt.DataType:
        dtypes = (value.dtype for value in self.values)
        return dt.Struct.from_tuples(zip(self.names, dtypes))
