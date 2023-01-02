"""Encapsulation of SQL window frames."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import toolz

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis import util
from ibis.common.exceptions import IbisInputError
from ibis.common.grounds import Comparable


def _sequence_to_tuple(x):
    return tuple(x) if util.is_iterable(x) else x


class RowsWithMaxLookback(NamedTuple):
    rows: int | np.integer
    max_lookback: ir.IntervalValue


def _choose_non_empty_val(first, second):
    if isinstance(first, (int, np.integer)) and first:
        non_empty_value = first
    elif not isinstance(first, (int, np.integer)) and first is not None:
        non_empty_value = first
    else:
        non_empty_value = second
    return non_empty_value


def _determine_how(preceding):
    offset_type = type(get_preceding_value(preceding))
    if issubclass(offset_type, (int, np.integer)):
        how = 'rows'
    elif issubclass(offset_type, ir.IntervalScalar):
        how = 'range'
    else:
        raise TypeError(
            'Type {} is not supported for row- or range- based trailing '
            'window operations'.format(offset_type)
        )
    return how


@functools.singledispatch
def get_preceding_value(preceding):
    raise TypeError(
        "Type {} is not a valid type for 'preceding' "
        "parameter".format(type(preceding))
    )


@get_preceding_value.register(tuple)
def get_preceding_value_tuple(preceding):
    start, end = preceding
    if start is None:
        preceding_value = end
    else:
        preceding_value = start
    return preceding_value


@get_preceding_value.register(int)
@get_preceding_value.register(np.integer)
@get_preceding_value.register(ir.IntervalScalar)
def get_preceding_value_simple(preceding):
    return preceding


@get_preceding_value.register(RowsWithMaxLookback)
def get_preceding_value_mlb(preceding: RowsWithMaxLookback):
    preceding_value = preceding.rows
    if not isinstance(preceding_value, (int, np.integer)):
        raise TypeError(
            f"{type(preceding).__name__} only supports integer row-based indexing."
        )
    return preceding_value


class Window(Comparable):
    """A window frame.

    Notes
    -----
    This class is patterned after SQL window frame clauses.

    Using `None` for `preceding` or `following` indicates an unbounded frame.

    Use 0 for `CURRENT ROW`.
    """

    __slots__ = (
        '_group_by',
        '_order_by',
        '_hash',
        'preceding',
        'following',
        'max_lookback',
        'how',
    )

    def __init__(
        self,
        group_by=None,
        order_by=None,
        preceding=None,
        following=None,
        max_lookback=None,
        how='rows',
    ):
        self._group_by = tuple(
            toolz.unique(
                arg.op() if isinstance(arg, ir.Expr) else arg
                for arg in util.promote_list(group_by)
            )
        )
        self._order_by = tuple(
            toolz.unique(
                arg.op() if isinstance(arg, ir.Expr) else arg
                for arg in util.promote_list(order_by)
            )
        )

        if isinstance(preceding, RowsWithMaxLookback):
            # the offset interval is used as the 'preceding' value of a window
            # while 'rows' is used to adjust the window created using offset
            self.preceding = preceding.max_lookback
            self.max_lookback = preceding.rows
        else:
            self.preceding = _sequence_to_tuple(preceding)
            self.max_lookback = max_lookback

        self.following = _sequence_to_tuple(following)
        self.how = how

        self._validate_frame()
        self._hash = self._compute_hash()

    def _compute_hash(self) -> int:
        return hash(
            (
                *self._group_by,
                *self._order_by,
                (
                    self.preceding.op()
                    if isinstance(self.preceding, ir.Expr)
                    else self.preceding
                ),
                (
                    self.following.op()
                    if isinstance(self.following, ir.Expr)
                    else self.following
                ),
                self.how,
                self.max_lookback,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def _validate_frame(self):
        preceding_tuple = has_preceding = False
        following_tuple = has_following = False
        if self.preceding is not None:
            preceding_tuple = isinstance(self.preceding, tuple)
            has_preceding = True

        if self.following is not None:
            following_tuple = isinstance(self.following, tuple)
            has_following = True

        if (preceding_tuple and has_following) or (following_tuple and has_preceding):
            raise IbisInputError(
                'Can only specify one window side when you want an off-center window'
            )
        elif preceding_tuple:
            start, end = self.preceding
            if end is None:
                raise IbisInputError("preceding end point cannot be None")
            if end < 0:
                raise IbisInputError("preceding end point must be non-negative")
            if start is not None:
                if start < 0:
                    raise IbisInputError("preceding start point must be non-negative")
                if start <= end:
                    raise IbisInputError(
                        "preceding start must be greater than preceding end"
                    )
        elif following_tuple:
            start, end = self.following
            if start is None:
                raise IbisInputError("following start point cannot be None")
            if start < 0:
                raise IbisInputError("following start point must be non-negative")
            if end is not None:
                if end < 0:
                    raise IbisInputError("following end point must be non-negative")
                if start >= end:
                    raise IbisInputError(
                        "following start must be less than following end"
                    )
        else:
            if not isinstance(self.preceding, ir.Expr):
                if has_preceding and self.preceding < 0:
                    raise IbisInputError(
                        f"'preceding' must be positive, got {self.preceding}"
                    )

            if not isinstance(self.following, ir.Expr):
                if has_following and self.following < 0:
                    raise IbisInputError(
                        f"'following' must be positive, got {self.following}"
                    )
        if self.how not in {'rows', 'range'}:
            raise IbisInputError(f"'how' must be 'rows' or 'range', got {self.how}")

        if self.max_lookback is not None:
            import pandas as pd

            if not isinstance(self.preceding, (ir.IntervalValue, pd.Timedelta)):
                raise IbisInputError(
                    "'max_lookback' must be specified as an interval "
                    "or pandas.Timedelta object"
                )

    def bind(self, table):
        # Internal API, ensure that any unresolved expr references (as strings,
        # say) are bound to the table being windowed

        groups = rlz.tuple_of(
            rlz.one_of((rlz.column_from(rlz.just(table)), rlz.any)),
            self._group_by,
        )
        sorts = rlz.tuple_of(rlz.sort_key_from(rlz.just(table)), self._order_by)

        return self._replace(group_by=groups, order_by=sorts)

    def combine(self, window):
        if self.how != window.how:
            raise IbisInputError(
                "Window types must match. "
                f"Expecting {self.how!r} window, got {window.how!r}"
            )

        return Window(
            preceding=_choose_non_empty_val(self.preceding, window.preceding),
            following=_choose_non_empty_val(self.following, window.following),
            group_by=self._group_by + window._group_by,
            order_by=self._order_by + window._order_by,
            max_lookback=self.max_lookback or window.max_lookback,
            how=self.how,
        )

    def group_by(self, expr):
        new_groups = self._group_by + tuple(util.promote_list(expr))
        return self._replace(group_by=new_groups)

    def _replace(self, **kwds):
        new_kwds = {
            'group_by': kwds.get('group_by', self._group_by),
            'order_by': kwds.get('order_by', self._order_by),
            'preceding': kwds.get('preceding', self.preceding),
            'following': kwds.get('following', self.following),
            'max_lookback': kwds.get('max_lookback', self.max_lookback),
            'how': kwds.get('how', self.how),
        }
        return Window(**new_kwds)

    def order_by(self, expr):
        new_sorts = self._order_by + tuple(util.promote_list(expr))
        return self._replace(order_by=new_sorts)

    def __equals__(self, other):
        return (
            self.max_lookback == other.max_lookback
            and (
                self.preceding.equals(other.preceding)
                if isinstance(self.preceding, ir.Expr)
                else self.preceding == other.preceding
            )
            and (
                self.following.equals(other.following)
                if isinstance(self.following, ir.Expr)
                else self.following == other.following
            )
            and self._group_by == other._group_by
            and self._order_by == other._order_by
        )

    def equals(self, other):
        if not isinstance(other, Window):
            raise TypeError(
                f"invalid equality comparison between {type(self)} and {type(other)}"
            )
        return self.__cached_equals__(other)


def rows_with_max_lookback(
    rows: int | np.integer,
    max_lookback: ir.IntervalValue,
) -> RowsWithMaxLookback:
    """Create a bound preceding value for use with trailing window functions.

    Parameters
    ----------
    rows
        Number of rows
    max_lookback
        Maximum lookback in time

    Returns
    -------
    RowsWithMaxLookback
        A named tuple of rows and maximum look-back in time
    """
    return RowsWithMaxLookback(rows, max_lookback)


def window(preceding=None, following=None, group_by=None, order_by=None):
    """Create a window clause for use with window functions.

    The `ROWS` window clause includes peer rows based on differences in row
    **number** whereas `RANGE` includes rows based on the differences in row
    **value** of a single `order_by` expression.

    All window frame bounds are inclusive.

    Parameters
    ----------
    preceding
        Number of preceding rows in the window
    following
        Number of following rows in the window
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return Window(
        preceding=preceding,
        following=following,
        group_by=group_by,
        order_by=order_by,
        how='rows',
    )


def range_window(preceding=None, following=None, group_by=None, order_by=None):
    """Create a range-based window clause for use with window functions.

    This RANGE window clause aggregates rows based upon differences in the
    value of the order-by expression.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    preceding
        Number of preceding rows in the window
    following
        Number of following rows in the window
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return Window(
        preceding=preceding,
        following=following,
        group_by=group_by,
        order_by=order_by,
        how='range',
    )


def cumulative_window(group_by=None, order_by=None) -> Window:
    """Create a cumulative window for use with window functions.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return Window(preceding=None, following=0, group_by=group_by, order_by=order_by)


def trailing_window(preceding, group_by=None, order_by=None):
    """Create a trailing window for use with aggregate window functions.

    Parameters
    ----------
    preceding
        The number of preceding rows
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    how = _determine_how(preceding)
    return Window(
        preceding=preceding,
        following=0,
        group_by=group_by,
        order_by=order_by,
        how=how,
    )


def trailing_range_window(preceding, order_by, group_by=None) -> Window:
    """Create a trailing range window for use with window functions.

    Parameters
    ----------
    preceding
        A value expression
    order_by
        Ordering key
    group_by
        Grouping key

    Returns
    -------
    Window
        A window frame
    """
    return Window(
        preceding=preceding,
        following=0,
        group_by=group_by,
        order_by=order_by,
        how='range',
    )


# TODO(kszucs): use ibis.expr.analysis.substitute instead
def propagate_down_window(node: ops.Node, window: Window):
    import ibis.expr.operations as ops

    clean_args = []
    unchanged = True
    for arg in node.args:
        if isinstance(arg, ops.Value) and not isinstance(node, ops.Window):
            new_arg = propagate_down_window(arg, window)
            if isinstance(new_arg, ops.Analytic):
                new_arg = ops.Window(new_arg, window)
            if arg is not new_arg:
                unchanged = False
            arg = new_arg

        clean_args.append(arg)

    if unchanged:
        return node
    else:
        return type(node)(*clean_args)
