from __future__ import absolute_import
from functools import partial

from multipledispatch import Dispatcher
import toolz

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
Given a node, compute a (possibly partial) scope prior to standard execution.

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
