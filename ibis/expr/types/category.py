from public import public

from ibis.expr.types.generic import Column, Scalar, Value


@public
class CategoryValue(Value):
    pass  # noqa: E701,E302


@public
class CategoryScalar(Scalar, CategoryValue):
    pass  # noqa: E701,E302


@public
class CategoryColumn(Column, CategoryValue):
    pass  # noqa: E701,E302
