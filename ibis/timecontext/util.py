""" Time context module

This is an attempt to allow us to implement the time context extension without
affecting the existing SQL-like execution model for backends.

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
from typing import Optional

import pandas as pd

import ibis.common.exceptions as com
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


def compare_timecontext(cur_context: TimeContext, old_context: TimeContext):
    """Compare two timecontext and return the relationship between two time
    context (SUBSET, SUPERSET, OVERLAP, NONOVERLAP).

    Parameters
    ----------
    cur_context: TimeContext
    old_context: TimeContext

    Returns
    -------
    result : Enum[TimeContextRelation]
    """
    begin, end = cur_context
    old_begin, old_end = old_context
    if old_begin <= begin and old_end >= end:
        return TimeContextRelation.SUBSET
    elif old_begin >= begin and old_end <= end:
        return TimeContextRelation.SUPERSET
    elif old_end < begin or end < old_begin:
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
