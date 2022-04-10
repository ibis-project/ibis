from public import public

from ibis.expr.types.generic import AnyColumn, AnyScalar, AnyValue


@public
class EnumValue(AnyValue):
    pass  # noqa: E701,E302


@public
class EnumScalar(AnyScalar, EnumValue):
    pass  # noqa: E701,E302


@public
class EnumColumn(AnyColumn, EnumValue):
    pass  # noqa: E701,E302


@public
class SetValue(AnyValue):
    pass  # noqa: E701,E302


@public
class SetScalar(AnyScalar, SetValue):
    pass  # noqa: E701,E302


@public
class SetColumn(AnyColumn, SetValue):
    pass  # noqa: E701,E302
