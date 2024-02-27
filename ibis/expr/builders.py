from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis import util
from ibis.common.annotations import annotated, attribute
from ibis.common.deferred import Deferred, Resolver, deferrable
from ibis.common.exceptions import IbisInputError
from ibis.common.grounds import Concrete
from ibis.common.typing import VarTuple  # noqa: TCH001

if TYPE_CHECKING:
    from typing_extensions import Self


class Builder(Concrete):
    pass


@deferrable(repr="<case>")
def _finish_searched_case(cases, results, default) -> ir.Value:
    """Finish constructing a SearchedCase expression.

    This is split out into a separate function to allow for deferred arguments
    to resolve.
    """
    return ops.SearchedCase(cases=cases, results=results, default=default).to_expr()


class SearchedCaseBuilder(Builder):
    """A case builder, used for constructing `ibis.case()` expressions."""

    cases: VarTuple[Union[Resolver, ops.Value[dt.Boolean]]] = ()
    results: VarTuple[Union[Resolver, ops.Value]] = ()
    default: Optional[Union[Resolver, ops.Value]] = None

    def when(self, case_expr: Any, result_expr: Any) -> Self:
        """Add a new condition and result to the `CASE` expression.

        Parameters
        ----------
        case_expr
            Predicate expression to use for this case.
        result_expr
            Value when the case predicate evaluates to true.

        """
        return self.copy(
            cases=self.cases + (case_expr,), results=self.results + (result_expr,)
        )

    def else_(self, result_expr: Any) -> Self:
        """Add a default value for the `CASE` expression.

        Parameters
        ----------
        result_expr
            Value to use when all case predicates evaluate to false.

        """
        return self.copy(default=result_expr)

    def end(self) -> ir.Value | Deferred:
        """Finish the `CASE` expression."""
        return _finish_searched_case(self.cases, self.results, self.default)


class SimpleCaseBuilder(Builder):
    """A case builder, used for constructing `Column.case()` expressions."""

    base: ops.Value
    cases: VarTuple[ops.Value] = ()
    results: VarTuple[ops.Value] = ()
    default: Optional[ops.Value] = None

    def when(self, case_expr: Any, result_expr: Any) -> Self:
        """Add a new condition and result to the `CASE` expression.

        Parameters
        ----------
        case_expr
            Expression to equality-compare with base expression. Must be
            comparable with the base.
        result_expr
            Value when the case predicate evaluates to true.

        """
        if not isinstance(case_expr, ir.Value):
            case_expr = ibis.literal(case_expr)
        if not isinstance(result_expr, ir.Value):
            result_expr = ibis.literal(result_expr)

        if not rlz.comparable(self.base, case_expr.op()):
            raise TypeError(
                f"Base expression {rlz._arg_type_error_format(self.base)} and "
                f"case {rlz._arg_type_error_format(case_expr)} are not comparable"
            )
        return self.copy(
            cases=self.cases + (case_expr,), results=self.results + (result_expr,)
        )

    def else_(self, result_expr: Any) -> Self:
        """Add a default value for the `CASE` expression.

        Parameters
        ----------
        result_expr
            Value to use when all case predicates evaluate to false.

        """
        return self.copy(default=result_expr)

    def end(self) -> ir.Value:
        """Finish the `CASE` expression."""
        if (default := self.default) is None:
            default = ibis.null().cast(rlz.highest_precedence_dtype(self.results))
        return ops.SimpleCase(
            cases=self.cases, results=self.results, default=default, base=self.base
        ).to_expr()


RowsWindowBoundary = ops.WindowBoundary[dt.Integer]
RangeWindowBoundary = ops.WindowBoundary[dt.Numeric | dt.Interval]


class WindowBuilder(Builder):
    """An unbound window frame specification.

    Notes
    -----
    This class is patterned after SQL window frame clauses.

    Using `None` for `preceding` or `following` indicates an unbounded frame.

    Use 0 for `CURRENT ROW`.

    """

    how: Literal["rows", "range"] = "rows"
    start: Optional[RangeWindowBoundary] = None
    end: Optional[RangeWindowBoundary] = None
    groupings: VarTuple[Union[str, Resolver, ops.Value]] = ()
    orderings: VarTuple[Union[str, Resolver, ops.Value]] = ()

    @attribute
    def _table(self):
        inputs = (
            self.start,
            self.end,
            *self.groupings,
            *self.orderings,
        )
        valuerels = (v.relations for v in inputs if isinstance(v, ops.Value))
        relations = frozenset().union(*valuerels)
        if len(relations) == 0:
            return None
        elif len(relations) == 1:
            (table,) = relations
            return table
        else:
            raise IbisInputError("Window frame can only depend on a single relation")

    def _maybe_cast_boundaries(self, start, end):
        if start and end:
            if start.dtype.is_interval() and end.dtype.is_numeric():
                return start, ops.Cast(end.value, start.dtype)
            elif start.dtype.is_numeric() and end.dtype.is_interval():
                return ops.Cast(start.value, end.dtype), end
        return start, end

    def _determine_how(self, start, end):
        if start and not start.dtype.is_integer():
            return self.range
        elif end and not end.dtype.is_integer():
            return self.range
        else:
            return self.rows

    def _validate_boundaries(self, start, end):
        start_, end_ = -math.inf, math.inf
        if start and isinstance(lit := start.value, ops.Literal):
            start_ = -lit.value if start.preceding else lit.value
        if end and isinstance(lit := end.value, ops.Literal):
            end_ = -lit.value if end.preceding else lit.value

        if start_ > end_:
            raise IbisInputError(
                "Window frame's start point must be greater than its end point"
            )

    @annotated
    def rows(
        self, start: Optional[RowsWindowBoundary], end: Optional[RowsWindowBoundary]
    ):
        self._validate_boundaries(start, end)
        start, end = self._maybe_cast_boundaries(start, end)
        return self.copy(how="rows", start=start, end=end)

    @annotated
    def range(
        self, start: Optional[RangeWindowBoundary], end: Optional[RangeWindowBoundary]
    ):
        self._validate_boundaries(start, end)
        start, end = self._maybe_cast_boundaries(start, end)
        return self.copy(how="range", start=start, end=end)

    @annotated
    def between(
        self, start: Optional[RangeWindowBoundary], end: Optional[RangeWindowBoundary]
    ):
        self._validate_boundaries(start, end)
        start, end = self._maybe_cast_boundaries(start, end)
        method = self._determine_how(start, end)
        return method(start, end)

    def group_by(self, expr) -> Self:
        return self.copy(groupings=self.groupings + util.promote_tuple(expr))

    def order_by(self, expr) -> Self:
        return self.copy(orderings=self.orderings + util.promote_tuple(expr))

    @annotated
    def bind(self, table: Optional[ops.Relation]):
        table = table or self._table
        if table is None:
            raise IbisInputError("Unable to bind window frame to a table")

        table = table.to_expr()

        def bind_value(value):
            if isinstance(value, str):
                return table._get_column(value)
            elif isinstance(value, Resolver):
                return value.resolve({"_": table})
            else:
                return value

        groupings = map(bind_value, self.groupings)
        orderings = map(bind_value, self.orderings)
        if self.how == "rows":
            return ops.RowsWindowFrame(
                table=table,
                start=self.start,
                end=self.end,
                group_by=groupings,
                order_by=orderings,
            )
        elif self.how == "range":
            return ops.RangeWindowFrame(
                table=table,
                start=self.start,
                end=self.end,
                group_by=groupings,
                order_by=orderings,
            )
        else:
            raise ValueError(f"Unsupported `{self.how}` window type")


class LegacyWindowBuilder(WindowBuilder):
    def _is_negative(self, value):
        if value is None:
            return False
        if isinstance(value, ir.Scalar):
            value = value.op().value
        return value < 0

    def preceding_following(self, preceding, following, how=None) -> Self:
        preceding_tuple = has_preceding = False
        following_tuple = has_following = False
        if preceding is not None:
            preceding_tuple = isinstance(preceding, tuple)
            has_preceding = True
        if following is not None:
            following_tuple = isinstance(following, tuple)
            has_following = True

        if (preceding_tuple and has_following) or (following_tuple and has_preceding):
            raise IbisInputError(
                "Can only specify one window side when you want an off-center window"
            )
        elif preceding_tuple:
            start, end = preceding
            if end is None:
                raise IbisInputError("preceding end point cannot be None")
            elif self._is_negative(end):
                raise IbisInputError("preceding end point must be non-negative")
            elif self._is_negative(start):
                raise IbisInputError("preceding start point must be non-negative")
            between = (
                None if start is None else ops.WindowBoundary(start, preceding=True),
                ops.WindowBoundary(end, preceding=True),
            )
        elif following_tuple:
            start, end = following
            if start is None:
                raise IbisInputError("following start point cannot be None")
            elif self._is_negative(start):
                raise IbisInputError("following start point must be non-negative")
            elif self._is_negative(end):
                raise IbisInputError("following end point must be non-negative")
            between = (
                ops.WindowBoundary(start, preceding=False),
                None if end is None else ops.WindowBoundary(end, preceding=False),
            )
        elif has_preceding and has_following:
            between = (
                ops.WindowBoundary(preceding, preceding=True),
                ops.WindowBoundary(following, preceding=False),
            )
        elif has_preceding:
            if self._is_negative(preceding):
                raise IbisInputError("preceding end point must be non-negative")
            between = (ops.WindowBoundary(preceding, preceding=True), None)
        elif has_following:
            if self._is_negative(following):
                raise IbisInputError("following end point must be non-negative")
            between = (None, ops.WindowBoundary(following, preceding=False))

        if how is None:
            return self.between(*between)
        elif how == "rows":
            return self.rows(*between)
        elif how == "range":
            return self.range(*between)
        else:
            raise ValueError(f"Invalid window frame type: {how}")
