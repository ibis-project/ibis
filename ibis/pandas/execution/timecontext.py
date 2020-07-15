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

Note: In order to use the feature we implemented here, there should be a
column of Timestamp type, and named as 'time' in TableExpr.
See ``execute_database_table_client`` in ``generic.py``.
And we assume timecontext is passed in as a tuple (begin, end) where begin and
end are timestamp, or datetime string like "20100101".
"""

import pandas as pd

import ibis.expr.operations as ops
from ibis.pandas.dispatch import compute_time_context, is_computable_input


@compute_time_context.register(ops.AsOfJoin)
def adjust_context_asof_join(op, scope=None, timecontext=None, **kwargs):
    new_timecontexts = [
        timecontext for arg in op.inputs if is_computable_input(arg)
    ]
    # right table should look back or forward
    try:
        begin, end = map(pd.to_datetime, timecontext)
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
    except ValueError:
        pass
    finally:
        return new_timecontexts
