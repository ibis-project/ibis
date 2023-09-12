"""Time context module.

This is an implementation of time context extension without affecting the
existing SQL-like execution model for backends.

Most of the execution is built on the foundation that "Data is uniquely
defined by the op tree". This is true in SQL analysis where there is no
ambiguity what the result of executing a Table is.

In time series analysis, however, this is not necessarily True. We have defined
an extension to ibis execution for time series analysis where the result of
executing a Table is defined by the Table plus the time context are
associated with the execution.

Time context specifies the temporal range of a query, it carries the start and
end datetimes. For example, a Table can represent the query select count(a)
from table, but the result of that is different with time context
("20190101", "20200101") vs ("20200101", "20210101"), because what data is in
"table" depends also on the time context.

While data in scope is public and global for all nodes, `timecontext` is
intended to store 'local' time context data for each node in execution. i.e.,
each subtree of an expr tree can have different time context. Which makes it
so that when executing each node, we also need to know the "local time context"
for that node.

And we propose to store these data as 'timecontext', calculate in execution
pass it along to children nodes, in the ibis tree. See each backends for
implementation details.

Time context adjustment algorithm
    In an Ibis tree, time context is local for each node, and they should be
    adjusted accordingly for some specific nodes. Those operations may
    require extra data outside of the global time context that user defines.
    For example, in asof_join, we need to look back extra `tolerance` daays
    for the right table to get the data for joining. Similarly for window
    operation with preceding and following.
    Algorithm to calculate context adjustment are defined in this module
    and could be used by multiple backends.
"""

from __future__ import annotations

import enum
import functools
from typing import TYPE_CHECKING, Any

import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis import config

TimeContext = tuple[pd.Timestamp, pd.Timestamp]


if TYPE_CHECKING:
    from ibis.backends.base.df.scope import Scope


# In order to use time context feature, there must be a column of Timestamp
# type, and named as 'time' in Table. This TIME_COL constant will be
# used in filtering data from a table or columns of a table. It can be changed
# by running:
#
# ibis.config.options.context_adjustment.time_col = "other_time_col"


def get_time_col():
    return config.options.context_adjustment.time_col


class TimeContextRelation(enum.Enum):
    """Enum to classify the relationship between two time contexts.

    Assume that we have two timecontext `c1 (begin1, end1)`, `c2(begin2, end2)`:

    - `SUBSET` means `c1` is a subset of `c2`, `begin1` is greater than or
      equal to `begin2`, and `end1` is less than or equal to `end2`.
    - `SUPERSET` means that `begin1` is earlier than `begin2`, and `end1`
      is later than `end2`.
    - If neither of the two contexts is a superset of each other, and they
      share some time range in common, we called them `OVERLAP`.
    - `NONOVERLAP` means the two contexts doesn't overlap at all, which
      means `end1` is earlier than `begin2` or `end2` is earlier than
      `begin1`.
    """

    SUBSET = 0
    SUPERSET = 1
    OVERLAP = 2
    NONOVERLAP = 3


def compare_timecontext(
    left_context: TimeContext, right_context: TimeContext
) -> TimeContextRelation:
    """Compare two time contexts and return the relationship between them."""
    left_begin, left_end = left_context
    right_begin, right_end = right_context
    if right_begin <= left_begin and right_end >= left_end:
        return TimeContextRelation.SUBSET
    elif right_begin >= left_begin and right_end <= left_end:
        return TimeContextRelation.SUPERSET
    elif right_end < left_begin or left_end < right_begin:
        return TimeContextRelation.NONOVERLAP
    else:
        return TimeContextRelation.OVERLAP


def canonicalize_context(
    timecontext: TimeContext | None,
) -> TimeContext | None:
    """Canonicalize a timecontext with type pandas.Timestamp for its begin and end time."""

    SUPPORTS_TIMESTAMP_TYPE = pd.Timestamp
    if not isinstance(timecontext, tuple) or len(timecontext) != 2:
        raise com.IbisError(f"Timecontext {timecontext} should specify (begin, end)")

    begin, end = timecontext

    if not isinstance(begin, SUPPORTS_TIMESTAMP_TYPE):
        raise com.IbisError(
            f"begin time value {begin} of type {type(begin)} is not"
            " of type pd.Timestamp"
        )
    if not isinstance(end, SUPPORTS_TIMESTAMP_TYPE):
        raise com.IbisError(
            f"end time value {end} of type {type(begin)} is not of type pd.Timestamp"
        )
    if begin > end:
        raise com.IbisError(
            f"begin time {begin} must be before or equal to end time {end}"
        )
    return begin, end


def localize_context(timecontext: TimeContext, timezone: str) -> TimeContext:
    """Localize tz-naive context."""
    begin, end = timecontext
    if begin.tz is None:
        begin = begin.tz_localize(timezone)

    if end.tz is None:
        end = end.tz_localize(timezone)

    return begin, end


def construct_time_context_aware_series(
    series: pd.Series, frame: pd.DataFrame
) -> pd.Series:
    """Construct a Series by adding 'time' in its MultiIndex.

    In window execution, the result Series of udf may need
    to be trimmed by timecontext. In order to do so, 'time'
    must be added as an index to the Series. We extract
    time column from the parent Dataframe `frame`.
    See `trim_window_result` in execution/window.py for
    trimming implementation.

    Examples
    --------
    >>> import pandas as pd
    >>> from ibis.backends.base.df.timecontext import (
    ...     construct_time_context_aware_series,
    ... )
    >>> df = pd.DataFrame(
    ...     {
    ...         "time": pd.Series(pd.date_range(start="2017-01-02", periods=3).values),
    ...         "id": [1, 2, 3],
    ...         "value": [1.1, 2.2, 3.3],
    ...     }
    ... )
    >>> df
            time  id  value
    0 2017-01-02   1    1.1
    1 2017-01-03   2    2.2
    2 2017-01-04   3    3.3
    >>> series = df["value"]
    >>> series
    0    1.1
    1    2.2
    2    3.3
    Name: value, dtype: float64
    >>> construct_time_context_aware_series(
    ...     series, df
    ... )  # quartodoc: +SKIP # doctest: +SKIP
       time
    0  2017-01-02    1.1
    1  2017-01-03    2.2
    2  2017-01-04    3.3
    Name: value, dtype: float64

    The index will be a MultiIndex of the original RangeIndex
    and a DateTimeIndex.

    >>> timed_series = construct_time_context_aware_series(series, df)
    >>> timed_series  # quartodoc: +SKIP # doctest: +SKIP
       time
    0  2017-01-02    1.1
    1  2017-01-03    2.2
    2  2017-01-04    3.3
    Name: value, dtype: float64

    >>> construct_time_context_aware_series(
    ...     timed_series, df
    ... )  # quartodoc: +SKIP # doctest: +SKIP
       time
    0  2017-01-02    1.1
    1  2017-01-03    2.2
    2  2017-01-04    3.3
    Name: value, dtype: float64
    The result is unchanged for a series already has 'time' as its index.
    """
    time_col = get_time_col()
    if time_col == frame.index.name:
        time_index = frame.index
    elif time_col in frame:
        time_index = pd.Index(frame[time_col])
    else:
        raise com.IbisError(f'"time" column not present in DataFrame {frame}')
    if time_col not in series.index.names:
        series.index = pd.MultiIndex.from_arrays(
            list(map(series.index.get_level_values, range(series.index.nlevels)))
            + [time_index],
            names=series.index.names + [time_col],
        )
    return series


@functools.singledispatch
def adjust_context(op: Any, scope: Scope, timecontext: TimeContext) -> TimeContext:
    """Adjust the `timecontext` for `op`.

    Parameters
    ----------
    op
        Ibis operation.
    scope
        Incoming scope.
    timecontext
        Time context associated with the node.

    Returns
    -------
    TimeContext
        For `op` that is not of type Node, raise an error to avoid failing
        silently since the default behavior is to return `timecontext`.
    """
    raise com.IbisError(f"Unsupported input type for adjust context for {op}")


@adjust_context.register(ops.Node)
def adjust_context_node(
    op: ops.Node, scope: Scope, timecontext: TimeContext
) -> TimeContext:
    # For any node, by default, do not adjust time context
    return timecontext


@adjust_context.register(ops.Alias)
def adjust_context_alias(
    op: ops.Node, scope: Scope, timecontext: TimeContext
) -> TimeContext:
    # For any node, by default, do not adjust time context
    return adjust_context(op.arg, scope, timecontext)


@adjust_context.register(ops.AsOfJoin)
def adjust_context_asof_join(
    op: ops.AsOfJoin, scope: Scope, timecontext: TimeContext
) -> TimeContext:
    begin, end = timecontext

    if op.tolerance is not None:
        from ibis.backends.pandas.execution import execute

        timedelta = execute(op.tolerance)
        return (begin - timedelta, end)

    return timecontext


@adjust_context.register(ops.WindowFunction)
def adjust_context_window(
    op: ops.WindowFunction, scope: Scope, timecontext: TimeContext
) -> TimeContext:
    # TODO(kszucs): this file should be really moved to the pandas
    # backend instead of the current central placement
    from ibis.backends.pandas.execution import execute

    # adjust time context by preceding and following
    begin, end = timecontext

    if op.frame.start is not None:
        value = execute(op.frame.start.value)
        if value:
            begin = begin - value

    if op.frame.end is not None:
        value = execute(op.frame.end.value)
        if value:
            end = end + value

    return (begin, end)
