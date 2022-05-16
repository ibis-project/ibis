from public import public

from ibis.expr.types.generic import AnyColumn, AnyScalar, Value


@public
class CategoryValue(Value):
    pass  # noqa: E701,E302


@public
class CategoryScalar(AnyScalar, CategoryValue):
    pass  # noqa: E701,E302


@public
class CategoryColumn(AnyColumn, CategoryValue):
    pass  # noqa: E701,E302
