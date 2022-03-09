from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import types as ir

from public import public

from .core import _binop
from .numeric import NumericColumn, NumericScalar, NumericValue


@public
class BooleanValue(NumericValue):
    def ifelse(
        self,
        true_expr: ir.ValueExpr,
        false_expr: ir.ValueExpr,
    ) -> ir.ValueExpr:
        """Construct a ternary conditional expression.

        Parameters
        ----------
        true_expr
            Expression to return if `self` evaluates to `True`
        false_expr
            Expression to return if `self` evaluates to `False`

        Returns
        -------
        ValueExpr
            The value of `true_expr` if `arg` is `True` else `false_expr`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([("is_person", "boolean")], name="t")
        >>> expr = t.is_person.ifelse("yes", "no")
        >>> print(ibis.impala.compile(expr))
        SELECT CASE WHEN t0.is_person THEN :param_1 ELSE :param_2 END AS tmp
        FROM t AS t0
        """
        import ibis

        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        return ibis.case().when(self, true_expr).else_(false_expr).end()

    def __and__(self, other: BooleanValue) -> BooleanValue:
        from .. import operations as ops
        from .. import rules as rlz

        return _binop(ops.And, self, rlz.any(other))

    __rand__ = __and__

    def __or__(self, other: BooleanValue) -> BooleanValue:
        from .. import operations as ops
        from .. import rules as rlz

        return _binop(ops.Or, self, rlz.any(other))

    __ror__ = __or__

    def __xor__(self, other: BooleanValue) -> BooleanValue:
        from .. import operations as ops
        from .. import rules as rlz

        return _binop(ops.Xor, self, rlz.any(other))

    __rxor__ = __xor__

    def __invert__(self, other: BooleanValue) -> BooleanValue:
        from .. import operations as ops
        from .. import rules as rlz

        return _binop(ops.Not, self, rlz.any(other))


@public
class BooleanScalar(NumericScalar, BooleanValue):
    pass  # noqa: E701,E302


@public
class BooleanColumn(NumericColumn, BooleanValue):
    def any(self) -> BooleanValue:
        from .. import operations as ops

        return ops.Any(self).to_expr()

    def notany(self) -> BooleanValue:
        from .. import operations as ops

        return ops.NotAny(self).to_expr()

    def all(self) -> BooleanScalar:
        from .. import operations as ops

        return ops.All(self).to_expr()

    def notall(self) -> BooleanScalar:
        from .. import operations as ops

        return ops.NotAll(self).to_expr()

    def cumany(self) -> BooleanColumn:
        from .. import operations as ops

        return ops.CumulativeAny(self).to_expr()

    def cumall(self) -> BooleanColumn:
        from .. import operations as ops

        return ops.CumulativeAll(self).to_expr()
