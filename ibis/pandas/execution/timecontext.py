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
end are timestamp, or datetime string like "20100101".

This is an optional feature. The result of executing an expression without time
context is conceptually the same as executing an expression with (-inf, inf)
time context.
"""

import pandas as pd

import ibis.expr.api as ir
import ibis.expr.operations as ops
from ibis.pandas.core import compute_time_context
from ibis.pandas.execution import execute


@compute_time_context.register(ops.AsOfJoin)
def adjust_context_asof_join(op, num_args, timecontext, **kwargs):
    new_timecontexts = [timecontext for i in range(num_args)]

    if not timecontext:
        return new_timecontexts

    # right table should look back or forward
    begin, end = timecontext
    tolerance = op.tolerance
    if tolerance is not None:
        timedelta = pd.Timedelta(-tolerance.op().right.op().value)
        if timedelta <= pd.Timedelta(0):
            new_begin = begin + timedelta
            new_end = end
        else:
            new_begin = begin
            new_end = end + timedelta
    # right table is the second node in children
    new_timecontexts[1] = (new_begin, new_end)
    return new_timecontexts


@compute_time_context.register(ops.WindowOp)
def adjust_context_window(op, num_args, timecontext, **kwargs):
    new_timecontexts = [timecontext for i in range(num_args)]

    if not timecontext:
        return new_timecontexts

    # adjust time context by preceding and following
    begin, end = timecontext
    result = [begin, end]
    preceding = op.window.preceding
    following = op.window.following
    if preceding is not None:
        if isinstance(preceding, ir.IntervalScalar):
            new_preceding = execute(preceding)
        else:
            new_preceding = preceding
        if new_preceding:
            result[0] = begin - new_preceding
    if following is not None:
        if isinstance(following, ir.IntervalScalar):
            new_following = execute(following)
        else:
            new_following = following
        if new_following:
            result[1] = end + new_following
    new_timecontexts = [result for i in range(num_args)]
    return new_timecontexts
