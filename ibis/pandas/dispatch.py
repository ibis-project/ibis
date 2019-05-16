from __future__ import absolute_import

from functools import partial

import pandas as pd
import toolz
from multipledispatch import Dispatcher

import ibis
import ibis.common as com
import ibis.expr.operations as ops

# Individual operation execution
execute_node = Dispatcher(
    'execute_node',
    doc=(
        'Execute an individual operation given the operation and its computed '
        'arguments'
    ),
)


@execute_node.register(ops.Node)
def execute_node_without_scope(node, **kwargs):
    raise com.UnboundExpressionError(
        (
            'Node of type {!r} has no data bound to it. '
            'You probably tried to execute an expression without a data '
            'source.'
        ).format(type(node).__name__)
    )


pre_execute = Dispatcher(
    'pre_execute',
    doc="""\
Given a node and zero or more clients, compute a partial scope prior to
execution.

Notes
-----
This function is useful if parts of the tree structure need to be executed at
the same time or if there are other reasons to need to interrupt the regular
depth-first traversal of the tree.
""",
)


# Default returns an empty scope
@pre_execute.register(ops.Node)
@pre_execute.register(ops.Node, ibis.client.Client)
def pre_execute_default(node, *clients, **kwargs):
    return {}


# Merge the results of all client pre-execution with scope
@pre_execute.register(ops.Node, [ibis.client.Client])
def pre_execute_multiple_clients(node, *clients, scope=None, **kwargs):
    return toolz.merge(
        scope, *map(partial(pre_execute, node, scope=scope, **kwargs), clients)
    )


execute_first = Dispatcher(
    "execute_first", doc="Execute code before any nodes have been evaluated."
)


@execute_first.register(ops.Node)
@execute_first.register(ops.Node, ibis.client.Client)
def execute_first_default(node, *clients, **kwargs):
    return {}


@execute_first.register(ops.Node, [ibis.client.Client])
def execute_first_multiple_clients(node, *clients, scope=None, **kwargs):
    return toolz.merge(
        scope,
        *map(partial(execute_first, node, scope=scope, **kwargs), clients),
    )


execute_literal = Dispatcher(
    'execute_literal',
    doc="""\
Special case literal execution to avoid the dispatching overhead of
``execute_node``.

Parameters
----------
op : ibis.expr.operations.Node
value : object
    The literal value of the object, e.g., int, float.
datatype : ibis.expr.datatypes.DataType
    Used to specialize on expressions whose underlying value is of a different
    type than its would-be type. For example, interval values are represented
    by an integer.
""",
)


post_execute = Dispatcher(
    'post_execute',
    doc="""\
Execute code on the result of a computation.

Parameters
----------
op : ibis.expr.operations.Node
    The operation that was just executed
data : object
    The result of the computation
""",
)


@post_execute.register(ops.Node, object)
def post_execute_default(op, data, **kwargs):
    return data


execute_last = Dispatcher(
    "execute_last", doc="Execute code after all nodes have been evaluated."
)


@execute_last.register(ops.Node, object)
def execute_last_default(_, result, **kwargs):
    """Return the input result."""
    return result


@execute_last.register(ops.Node, pd.DataFrame)
def execute_last_dataframe(op, result, **kwargs):
    """Reset the `result` :class:`~pandas.DataFrame`."""
    schema = op.to_expr().schema()
    df = result.reset_index()
    return df.loc[:, schema.names]


@execute_last.register(ops.Node, pd.Series)
def execute_last_series(_, result, **kwargs):
    """Reset the `result` :class:`~pandas.Series`."""
    return result.reset_index(drop=True)


execute = Dispatcher("execute")
