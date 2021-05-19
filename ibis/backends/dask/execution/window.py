import operator
from typing import Optional

import dask.dataframe as dd

import ibis.expr.operations as ops
import ibis.expr.window as win
from ibis.backends.pandas.execution.window import (
    _post_process_empty,
    _post_process_group_by,
    compute_time_context,
    get_aggcontext,
)
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..core import execute
from ..dispatch import execute_node, pre_execute


@execute_node.register(ops.WindowOp, dd.Series, win.Window)
def execute_window_op(
    op,
    data,
    window,
    scope: Scope = None,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    if not (window.preceding is None and window.following is None):
        raise NotImplementedError(
            "Window operations are unsuported in the dask backend"
        )

    operand = op.expr
    # pre execute "manually" here because otherwise we wouldn't pickup
    # relevant scope changes from the child operand since we're managing
    # execution of that by hand
    operand_op = operand.op()

    adjusted_timecontext = None
    if timecontext:
        arg_timecontexts = compute_time_context(
            op, timecontext=timecontext, clients=clients
        )
        # timecontext is the original time context required by parent node
        # of this WindowOp, while adjusted_timecontext is the adjusted context
        # of this Window, since we are doing a manual execution here, use
        # adjusted_timecontext in later execution phases
        adjusted_timecontext = arg_timecontexts[0]

    pre_executed_scope = pre_execute(
        operand_op,
        *clients,
        scope=scope,
        timecontext=adjusted_timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    scope = scope.merge_scope(pre_executed_scope)
    (root,) = op.root_tables()
    root_expr = root.to_expr()

    data = execute(
        root_expr,
        scope=scope,
        timecontext=adjusted_timecontext,
        clients=clients,
        aggcontext=aggcontext,
        **kwargs,
    )

    group_by = window._group_by
    grouping_keys = [
        key_op.name
        if isinstance(key_op, ops.TableColumn)
        else execute(
            key,
            scope=scope,
            clients=clients,
            timecontext=adjusted_timecontext,
            aggcontext=aggcontext,
            **kwargs,
        )
        for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    if group_by:
        source = data.groupby(grouping_keys, sort=False)
        post_process = _post_process_group_by
    else:
        source = data
        post_process = _post_process_empty

    new_scope = scope.merge_scopes(
        [
            Scope({t: source}, adjusted_timecontext)
            for t in operand.op().root_tables()
        ],
        overwrite=True,
    )

    aggcontext = get_aggcontext(
        window,
        scope=scope,
        operand=operand,
        parent=source,
        group_by=grouping_keys,
        order_by=[],
        **kwargs,
    )
    result = execute(
        operand,
        scope=new_scope,
        timecontext=adjusted_timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    result = post_process(
        result, data, [], grouping_keys, adjusted_timecontext,
    )

    return result
