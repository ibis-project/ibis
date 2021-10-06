""" Time context module

This is an implementation of time context extension without affecting the
existing SQL-like execution model for backends.

Most of the execution is built on the foundation that "Data is uniquely
defined by the op tree". This is true in SQL analysis where there is no
ambiguity what the result of executing a TableExpr is.

In time series analysis, however, this is not necessarily True. We have defined
an extension to ibis execution for time series analysis where the result of
executing a TableExpr is defined by the TableExpr plus the time context are
associated with the execution.

Time context specifies the temporal range of a query, it carries the start and
end datetimes. For example, a TableExpr can represent the query select count(a)
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
"""

import enum
import functools
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

import ibis.common.exceptions as com
import ibis.config as config
import ibis.expr.api as ir
import ibis.expr.operations as ops
from ibis.expr.operations import Node
from ibis.expr.typing import TimeContext

if TYPE_CHECKING:
    from ibis.expr.scope import Scope

# In order to use time context feature, there must be a column of Timestamp
# type, and named as 'time' in TableExpr. This TIME_COL constant will be
# used in filtering data from a table or columns of a table. It can be changed
# by ibis.set_option('time_col')


def get_time_col():
    return config.options.context_adjustment.time_col


class TimeContextRelation(enum.Enum):
    """Enum to classify the relationship between two time contexts
    Assume that we have two timecontext `c1 (begin1, end1)`,
    `c2(begin2, end2)`:
        - SUBSET means `c1` is a subset of `c2`, `begin1` is greater than or
          equal to `begin2`, and `end1` is less than or equal to `end2`.
        - SUPERSET means that `begin1` is earlier than `begin2`, and `end1`
          is later than `end2`.
        - If neither of the two contexts is a superset of each other, and they
          share some time range in common, we called them OVERLAP.
        - NONOVERLAP means the two contexts doesn't overlap at all, which
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
    """Compare two timecontext and return the relationship between two time
    context (SUBSET, SUPERSET, OVERLAP, NONOVERLAP).

    Parameters
    ----------
    left_context: TimeContext
    right_context: TimeContext

    Returns
    -------
    result : TimeContextRelation
    """
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
    timecontext: Optional[TimeContext],
) -> Optional[TimeContext]:
    """Convert a timecontext to canonical one with type pandas.Timestamp
    for its begin and end time. Raise Exception for illegal inputs
    """
    SUPPORTS_TIMESTAMP_TYPE = pd.Timestamp
    if not isinstance(timecontext, tuple) or len(timecontext) != 2:
        raise com.IbisError(
            f'Timecontext {timecontext} should specify (begin, end)'
        )

    begin, end = timecontext

    if not isinstance(begin, SUPPORTS_TIMESTAMP_TYPE):
        raise com.IbisError(
            f'begin time value {begin} of type {type(begin)} is not'
            ' of type pd.Timestamp'
        )
    if not isinstance(end, SUPPORTS_TIMESTAMP_TYPE):
        raise com.IbisError(
            f'end time value {end} of type {type(begin)} is not'
            ' of type pd.Timestamp'
        )
    if begin > end:
        raise com.IbisError(
            f'begin time {begin} must be before or equal' f' to end time {end}'
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
    """Construct a Series by adding 'time' in its MultiIndex

    In window execution, the result Series of udf may need
    to be trimmed by timecontext. In order to do so, 'time'
    must be added as an index to the Series. We extract
    time column from the parent Dataframe `frame`.
    See `trim_window_result` in execution/window.py for
    trimming implementation.

    Parameters
    ----------
    series: pd.Series, the result series of an udf execution
    frame: pd.DataFrame, the parent Dataframe of `series`

    Returns
    -------
    pd.Series

    Examples
    --------
    >>> import pandas as pd
    >>> from ibis.expr.timecontext import construct_time_context_aware_series
    >>> df = pd.DataFrame(
    ...     {
    ...         'time': pd.Series(
    ...             pd.date_range(
    ...                 start='2017-01-02', periods=3
    ...             ).values
    ...         ),
    ...         'id': [1,2,3],
    ...         'value': [1.1, 2.2, 3.3],
    ...     }
    ... )
    >>> df
            time  id  value
    0 2017-01-02   1    1.1
    1 2017-01-03   2    2.2
    2 2017-01-04   3    3.3
    >>> series = df['value']
    >>> series
    0    1.1
    1    2.2
    2    3.3
    Name: value, dtype: float64
    >>> construct_time_context_aware_series(series, df)
       time
    0  2017-01-02    1.1
    1  2017-01-03    2.2
    2  2017-01-04    3.3
    Name: value, dtype: float64

    The index will be a MultiIndex of the original RangeIndex
    and a DateTimeIndex.

    >>> timed_series = construct_time_context_aware_series(series, df)
    >>> timed_series
       time
    0  2017-01-02    1.1
    1  2017-01-03    2.2
    2  2017-01-04    3.3
    Name: value, dtype: float64

    >>> construct_time_context_aware_series(timed_series, df)
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
            list(
                map(series.index.get_level_values, range(series.index.nlevels))
            )
            + [time_index],
            names=series.index.names + [time_col],
        )
    return series


""" Time context adjustment algorithm
    In an Ibis tree, time context is local for each node, and they should be
    adjusted accordingly for some specific nodes. Those operations may
    require extra data outside of the global time context that user defines.
    For example, in asof_join, we need to look back extra `tolerance` daays
    for the right table to get the data for joining. Similarly for window
    operation with preceeding and following.
    Algorithm to calculate context adjustment are defined in this module
    and could be used by multiple backends.
"""


@functools.singledispatch
def adjust_context(
    op: Any, timecontext: TimeContext, scope: Optional['Scope'] = None
) -> TimeContext:
    """
    Params
    -------
    op: ibis.expr.operations.Node
    timecontext: TimeContext
        time context associated with the node
    scope: Scope

    Returns
    --------
    Adjusted time context
        For op that is not of type Node, we raise an error to avoid failing
        silently since the default behavior is to return input timecontext
        itself.
    """
    raise com.IbisError(f'Unsupported input type for adjust context for {op}')


@adjust_context.register(ops.Node)
def adjust_context_node(
    op: Node, timecontext: TimeContext, scope: Optional['Scope'] = None
) -> TimeContext:
    # For any node, by default, do not adjust time context
    return timecontext


@adjust_context.register(ops.AsOfJoin)
def adjust_context_asof_join(
    op: ops.AsOfJoin, timecontext: TimeContext, scope: Optional['Scope'] = None
) -> TimeContext:
    begin, end = timecontext

    if op.tolerance is not None:
        from ibis.backends.pandas.execution import execute

        timedelta = execute(op.tolerance)
        return (begin - timedelta, end)

    return timecontext


@adjust_context.register(ops.WindowOp)
def adjust_context_window(
    op: ops.WindowOp, timecontext: TimeContext, scope: Optional['Scope'] = None
) -> TimeContext:
    # adjust time context by preceding and following
    begin, end = timecontext

    preceding = op.window.preceding
    if preceding is not None:
        if isinstance(preceding, ir.IntervalScalar):
            from ibis.backends.pandas.execution import execute

            preceding = execute(preceding)
        if preceding and not isinstance(preceding, (int, np.integer)):
            begin = begin - preceding

    following = op.window.following
    if following is not None:
        if isinstance(following, ir.IntervalScalar):
            from ibis.backends.pandas.execution import execute

            following = execute(following)
        if following and not isinstance(following, (int, np.integer)):
            end = end + following

    return (begin, end)
