from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.core import _binop
from ibis.expr.types.numeric import NumericColumn, NumericScalar, NumericValue

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class BooleanValue(NumericValue):
    def ifelse(
        self,
        true_expr: ir.Value,
        false_expr: ir.Value,
    ) -> ir.Value:
        """Construct a ternary conditional expression.

        Parameters
        ----------
        true_expr
            Expression to return if `self` evaluates to `True`
        false_expr
            Expression to return if `self` evaluates to `False`

        Returns
        -------
        Value
            The value of `true_expr` if `arg` is `True` else `false_expr`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([("is_person", "boolean")], name="t")
        >>> expr = t.is_person.ifelse("yes", "no")
        >>> print(ibis.impala.compile(expr.name("tmp")))
        SELECT if(t0.`is_person`, 'yes', 'no') AS `tmp`
        FROM t t0
        """
        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        return ops.Where(self, true_expr, false_expr).to_expr()

    def __and__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.And, self, other)

    __rand__ = __and__

    def __or__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.Or, self, other)

    __ror__ = __or__

    def __xor__(self, other: BooleanValue) -> BooleanValue:
        return _binop(ops.Xor, self, other)

    __rxor__ = __xor__

    def __invert__(self) -> BooleanValue:
        return self.negate()

    @staticmethod
    def __negate_op__():
        return ops.Not


@public
class BooleanScalar(NumericScalar, BooleanValue):
    pass


@public
class BooleanColumn(NumericColumn, BooleanValue):
    def any(self, where: BooleanValue | None = None) -> BooleanValue:
        """Return whether at least one element is `True`.

        Parameters
        ----------
        where
            Optional filter for the aggregation
        """
        import ibis.expr.analysis as an

        return an._make_any(self, ops.Any, where=where)

    def notany(self, where: BooleanValue | None = None) -> BooleanValue:
        """Return whether no elements are `True`."""
        import ibis.expr.analysis as an

        return an._make_any(self, ops.NotAny, where=where)

    def all(self, where: BooleanValue | None = None) -> BooleanScalar:
        """Return whether all elements are `True`."""
        return ops.All(self, where=where).to_expr()

    def notall(self, where: BooleanValue | None = None) -> BooleanScalar:
        """Return whether not all elements are `True`."""
        return ops.NotAll(self, where=where).to_expr()

    def cumany(self) -> BooleanColumn:
        """Accumulate the `any` aggregate."""
        return ops.CumulativeAny(self).to_expr()

    def cumall(self) -> BooleanColumn:
        """Accumulate the `all` aggregate."""
        return ops.CumulativeAll(self).to_expr()
