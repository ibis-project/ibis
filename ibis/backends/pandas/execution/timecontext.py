"""Implementation of compute_time_context for time context related operations.

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
column of Timestamp type, and named as 'time' in Table. And this 'time'
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
from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.expr.operations as ops
from ibis.backends.base.df.timecontext import TimeContext, adjust_context
from ibis.backends.pandas.core import (
    compute_time_context,
    get_node_arguments,
    is_computable_input,
)

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend
    from ibis.backends.base.df.scope import Scope


@compute_time_context.register(ops.AsOfJoin)
def compute_time_context_asof_join(
    op: ops.AsOfJoin,
    scope: Scope,
    clients: list[BaseBackend],
    timecontext: TimeContext | None = None,
    **kwargs,
):
    new_timecontexts = [
        timecontext for arg in get_node_arguments(op) if is_computable_input(arg)
    ]

    if not timecontext:
        return new_timecontexts

    # right table is the second node in children
    new_timecontexts = [
        new_timecontexts[0],
        adjust_context(op, scope, timecontext),
        *new_timecontexts[2:],
    ]
    return new_timecontexts


@compute_time_context.register(ops.Window)
def compute_time_context_window(
    op: ops.Window,
    scope: Scope,
    clients: list[BaseBackend],
    timecontext: TimeContext | None = None,
    **kwargs,
):
    new_timecontexts = [
        timecontext for arg in get_node_arguments(op) if is_computable_input(arg)
    ]

    if not timecontext:
        return new_timecontexts

    result = adjust_context(op, scope, timecontext)

    new_timecontexts = [
        result for arg in get_node_arguments(op) if is_computable_input(arg)
    ]
    return new_timecontexts
