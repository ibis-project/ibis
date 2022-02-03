from public import public

from .numeric import NumericColumn, NumericScalar, NumericValue


@public
class BooleanValue(NumericValue):
    pass  # noqa: E701,E302


@public
class BooleanScalar(NumericScalar, BooleanValue):
    pass  # noqa: E701,E302


@public
class BooleanColumn(NumericColumn, BooleanValue):
    pass  # noqa: E701,E302
