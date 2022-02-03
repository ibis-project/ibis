from public import public

from .generic import AnyColumn, AnyScalar, AnyValue


@public
class NumericValue(AnyValue):
    pass  # noqa: E701,E302


@public
class NumericScalar(AnyScalar, NumericValue):
    pass  # noqa: E701,E302


@public
class NumericColumn(AnyColumn, NumericValue):
    pass  # noqa: E701,E302


@public
class IntegerValue(NumericValue):
    pass  # noqa: E701,E302


@public
class IntegerScalar(NumericScalar, IntegerValue):
    pass  # noqa: E701,E302


@public
class IntegerColumn(NumericColumn, IntegerValue):
    pass  # noqa: E701,E302


@public
class FloatingValue(NumericValue):
    pass  # noqa: E701,E302


@public
class FloatingScalar(NumericScalar, FloatingValue):
    pass  # noqa: E701,E302


@public
class FloatingColumn(NumericColumn, FloatingValue):
    pass  # noqa: E701,E302


@public
class DecimalValue(NumericValue):
    pass  # noqa: E701,E302


@public
class DecimalScalar(NumericScalar, DecimalValue):
    pass  # noqa: E701,E302


@public
class DecimalColumn(NumericColumn, DecimalValue):
    pass  # noqa: E701,E302
