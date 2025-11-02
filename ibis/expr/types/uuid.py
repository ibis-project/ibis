from __future__ import annotations

from public import public

from ibis.expr import datatypes as dt
from ibis.expr.types.generic import Column, Scalar, Value


@public
class UUIDValue(Value):
    __dtype__ = dt.uuid


@public
class UUIDScalar(Scalar, UUIDValue):
    pass


@public
class UUIDColumn(Column, UUIDValue):
    pass
