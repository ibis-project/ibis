from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import ValidationError, attribute
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Value


@public
class StructField(Value):
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
    names: VarTuple[str]
    values: Optional[VarTuple[Value]]
    dtype: Optional[dt.Struct] = None

    def __init__(
        self,
        names: VarTuple[str],
        values: None | VarTuple[Value],
        dtype: dt.Struct | None = None,
    ):
        if len(names) == 0:
            raise ValidationError("StructColumn must have at least one field")
        if values is None:
            if dtype is None:
                raise ValidationError("If values is None, dtype must be provided")
            if not isinstance(dtype, dt.Struct):
                raise ValidationError(f"dtype must be a struct, got {dtype}")
        else:
            if len(names) != len(values):
                raise ValidationError(
                    f"Length of names ({len(names)}) does not match length of "
                    f"values ({len(values)})"
                )
            if dtype is None:
                dtype = dt.Struct.from_tuples(zip(names, (v.dtype for v in values)))
        super().__init__(names=names, values=values, dtype=dtype)

    @attribute
    def shape(self) -> ds.DataShape:
        if self.values is None:
            return ds.scalar
        return rlz.highest_precedence_shape(self.values)
