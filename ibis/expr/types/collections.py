from __future__ import annotations

from public import public

from ibis.expr.types.generic import Column, Scalar, Value


@public
class SetValue(Value):
    pass


@public
class SetScalar(Scalar, SetValue):
    pass


@public
class SetColumn(Column, SetValue):
    pass
