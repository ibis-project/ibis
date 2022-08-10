from typing import Optional, Sequence

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.expr.types.strings import StringValue


@public
class CategoryValue(Value):
    def label(
        self, labels: Sequence[str], nulls: Optional[str] = None
    ) -> StringValue:
        """Format a known number of categories as strings.

        Parameters
        ----------
        labels
            Labels to use for formatting categories
        nulls
            How to label any null values among the categories

        Returns
        -------
        StringValue
            Labeled categories
        """
        op = ops.CategoryLabel(self, labels, nulls)
        return op.to_expr()


@public
class CategoryScalar(Scalar, CategoryValue):
    pass  # noqa: E701,E302


@public
class CategoryColumn(Column, CategoryValue):
    pass  # noqa: E701,E302
