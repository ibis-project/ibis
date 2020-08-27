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

While data in scope is public and global for all nodes, timecontext is intended
to store 'local' time context data for each node in execution. i.e., each
subtree of an expr tree can have different time context. Which makes it so
that when executing each node, we also need to know the "local time context"
for that node.

And we propose to store these data as 'timecontext', calculate in execution
pass it along to children nodes, in the ibis tree. See each backends for
implementation details.
"""

import enum
import functools
from typing import Optional

import numpy as np
import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.api as ir
import ibis.expr.operations as ops
from ibis.client import Backends
from ibis.expr.operations import Node
from ibis.expr.typing import TimeContext

# In order to use time context feature, there must be a column of Timestamp
# type, and named as 'time' in TableExpr. This TIME_COL constant will be
# used in filtering data from a table or columns of a table.
TIME_COL = 'time'


class TimeContextRelation(enum.Enum):
    """ Enum to classify the relationship between two time contexts
    Assume that we have two timecontext c1 (begin1, end1),
    c2(begin2, end2):
    SUBSET means c1 is a subset of c2, begin1 is greater than or equal to
    begin2, and end1 is less than or equal to end2.
    Likewise, SUPERSET means that begin1 is earlier than begin2, and end1
    is later than end2.
    If neither of the two contexts is a superset of each other, and they
    share some time range in common, we called them OVERLAP.
    And NONOVERLAP means the two contexts doesn't overlap at all, which
    means end1 is earlier than begin2 or end2 is earlier than begin1
    """

    SUBSET = 0
    SUPERSET = 1
    OVERLAP = 2
    NONOVERLAP = 3


def compare_timecontext(left_context: TimeContext, right_context: TimeContext):
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
    try:
        begin, end = timecontext
    except (ValueError, TypeError):
        raise com.IbisError(
            f'Timecontext {timecontext} should specify (begin, end)'
        )

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
    op: Node, *clients: Backends, timecontext: TimeContext
) -> TimeContext:
    """
    Params
    -------
    op: ibis.expr.operations.Node
    clients: List[ibis.client.Client], backend for execution
    timecontext: TimeContext, time context associated with the node

    Returns
    --------
    Adjusted time context
    """
    # by default, do not adjust time context
    return timecontext


@adjust_context.register(ops.AsOfJoin)
def adjust_context_asof_join(
    op: Node, *clients: Backends, timecontext: TimeContext
) -> TimeContext:
    begin, end = timecontext

    if op.tolerance is not None:
        for backend in clients:
            try:
                timedelta = backend.execute(op.tolerance)
                # only backwards and adjust begin time only
                return (begin - timedelta, end)
            except Exception:
                pass

    return timecontext


@adjust_context.register(ops.WindowOp)
def adjust_context_window(
    op: Node, *clients: Backends, timecontext: TimeContext
) -> TimeContext:
    # adjust time context by preceding and following
    begin, end = timecontext

    preceding = op.window.preceding
    if preceding is not None:
        if isinstance(preceding, ir.IntervalScalar):
            for backend in clients:
                try:
                    preceding = backend.execute(preceding)
                    if preceding:
                        break
                except Exception:
                    pass
        if preceding and not isinstance(preceding, (int, np.integer)):
            begin = begin - preceding

    following = op.window.following
    if following is not None:
        if isinstance(following, ir.IntervalScalar):
            for backend in clients:
                try:
                    following = backend.execute(following)
                    if following:
                        break
                except Exception:
                    pass
        if following and not isinstance(following, (int, np.integer)):
            end = end + following

    return (begin, end)
