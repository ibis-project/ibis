from public import public

from ibis.expr.types.generic import AnyColumn, AnyScalar, AnyValue


@public
class CategoryValue(AnyValue):
    pass  # noqa: E701,E302


@public
class CategoryScalar(AnyScalar, CategoryValue):
    pass  # noqa: E701,E302


@public
class CategoryColumn(AnyColumn, CategoryValue):
    pass  # noqa: E701,E302
