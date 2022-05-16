from public import public

from ibis.expr.types.generic import AnyColumn, Scalar, Value


@public
class EnumValue(Value):
    pass  # noqa: E701,E302


@public
class EnumScalar(Scalar, EnumValue):
    pass  # noqa: E701,E302


@public
class EnumColumn(AnyColumn, EnumValue):
    pass  # noqa: E701,E302


@public
class SetValue(Value):
    pass  # noqa: E701,E302


@public
class SetScalar(Scalar, SetValue):
    pass  # noqa: E701,E302


@public
class SetColumn(AnyColumn, SetValue):
    pass  # noqa: E701,E302
