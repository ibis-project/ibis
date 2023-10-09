from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional, Union

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.common.annotations import annotated
from ibis.common.exceptions import IbisInputError
from ibis.common.grounds import Concrete
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.deferred import Deferred  # noqa: TCH001
from ibis.expr.operations.relations import Relation  # noqa: TCH001
from ibis.expr.types.relations import bind_expr

if TYPE_CHECKING:
    from typing_extensions import Self


RowsWindowBoundary = ops.WindowBoundary[dt.Integer]
RangeWindowBoundary = ops.WindowBoundary[dt.Numeric | dt.Interval]


class WindowBuilder(Concrete):
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
    groupings: VarTuple[Union[str, Deferred, ops.Value]] = ()
    orderings: VarTuple[Union[str, Deferred, ops.Value]] = ()
    max_lookback: Optional[ops.Value[dt.Interval]] = None

    def _maybe_cast_boundary(self, boundary, dtype):
        if boundary.dtype == dtype:
            return boundary
        value = ops.Cast(boundary.value, dtype)
        return boundary.copy(value=value)

    def _maybe_cast_boundaries(self, start, end):
        if start and end:
            dtype = dt.higher_precedence(start.dtype, end.dtype)
            start = self._maybe_cast_boundary(start, dtype)
            end = self._maybe_cast_boundary(end, dtype)
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

    def lookback(self, value) -> Self:
        return self.copy(max_lookback=value)

    @annotated
    def bind(self, table: Relation):
        groupings = bind_expr(table.to_expr(), self.groupings)
        orderings = bind_expr(table.to_expr(), self.orderings)
        if self.how == "rows":
            return ops.RowsWindowFrame(
                table=table,
                start=self.start,
                end=self.end,
                group_by=groupings,
                order_by=orderings,
                max_lookback=self.max_lookback,
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
