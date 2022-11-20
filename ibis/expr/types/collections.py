from public import public

from ibis.expr.types.generic import Column, Scalar, Value


@public
class SetValue(Value):
    pass  # noqa: E701,E302


@public
class SetScalar(Scalar, SetValue):
    pass  # noqa: E701,E302


@public
class SetColumn(Column, SetValue):
    pass  # noqa: E701,E302
