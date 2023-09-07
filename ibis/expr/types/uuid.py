from __future__ import annotations

from public import public

from ibis.expr.types.generic import Column, Scalar, Value


@public
class UUIDValue(Value):
    pass


@public
class UUIDScalar(Scalar, UUIDValue):
    pass


@public
class UUIDColumn(Column, UUIDValue):
    pass
