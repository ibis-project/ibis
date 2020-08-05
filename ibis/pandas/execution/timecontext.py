""" Implementation of compute_time_context for time context related operations

Time context of a node is computed at the beginning of execution phase.

To use time context to load time series data:

For operations like window, asof_join that adjust time context in execution,
implement ``compute_time_context`` to pass different time contexts to child
nodes.

If ``pre_execute`` preloads any data, it should use timecontext to trim data
to be in the time range.

``execute_node`` of a leaf node can use timecontext to trim data, or to pass
it as a filter in the database query.

In some cases, data need to be trimmed in ``post_execute``.

Note: In order to use the feature we implemented here, there must be a
column of Timestamp type, and named as 'time' in TableExpr. And this 'time'
column should be preserved across the expression tree. If 'time' column is
dropped then execution will result in error.
See ``execute_database_table_client`` in ``generic.py``.
And we assume timecontext is passed in as a tuple (begin, end) where begin and
end are timestamp, or datetime string like "20100101". Time range is inclusive
(include both begin and end points).

This is an optional feature. The result of executing an expression without time
context is conceptually the same as executing an expression with (-inf, inf)
time context.
"""
from typing import Optional

import ibis.expr.api as ir
import ibis.expr.operations as ops
from ibis.expr.typing import TimeContext
from ibis.pandas.core import compute_time_context, is_computable_input
from ibis.pandas.execution import execute
from ibis.timecontext.adjustment import (
    adjust_context_asof_join,
    adjust_context_window,
)


@compute_time_context.register(ops.AsOfJoin)
def compute_time_context_asof_join(
    op, timecontext: Optional[TimeContext], **kwargs
):
    new_timecontexts = [
        timecontext for arg in op.inputs if is_computable_input(arg)
    ]

    if not timecontext:
        return new_timecontexts

    if op.tolerance is not None:
        timedelta = execute(op.tolerance)
        result = adjust_context_asof_join(timecontext, timedelta)
    else:
        result = timecontext
    # right table is the second node in children
    new_timecontexts[1] = result
    return new_timecontexts


@compute_time_context.register(ops.WindowOp)
def compute_time_context_window(
    op, timecontext: Optional[TimeContext], **kwargs
):
    new_timecontexts = [
        timecontext for arg in op.inputs if is_computable_input(arg)
    ]

    if not timecontext:
        return new_timecontexts

    # adjust time context by preceding and following
    preceding = op.window.preceding
    if preceding is not None:
        if isinstance(preceding, ir.IntervalScalar):
            preceding = execute(preceding)

    following = op.window.following
    if following is not None:
        if isinstance(following, ir.IntervalScalar):
            following = execute(following)

    result = adjust_context_window(
        timecontext, preceding=preceding, following=following
    )

    new_timecontexts = [
        result for arg in op.inputs if is_computable_input(arg)
    ]
    return new_timecontexts
