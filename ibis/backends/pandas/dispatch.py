from __future__ import annotations

from functools import partial

from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base import BaseBackend
from ibis.backends.base.df.scope import Scope
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher

# Individual operation execution
execute_node = TraceTwoLevelDispatcher(
    "execute_node",
    doc=(
        "Execute an individual operation given the operation and its computed "
        "arguments"
    ),
)


@execute_node.register(ops.Node, [object])
def raise_unknown_op(node, *args, **kwargs):
    signature = ", ".join(type(arg).__name__ for arg in args)
    raise com.OperationNotDefinedError(
        "Operation is not implemented for this backend with "
        f"signature: execute_node({type(node).__name__}, {signature})"
    )


@execute_node.register(ops.TableNode)
def raise_unknown_table_node(node, **kwargs):
    raise com.UnboundExpressionError(
        f"Node of type {type(node).__name__!r} has no data bound to it. "
        "You probably tried to execute an expression without a data "
        "source."
    )


pre_execute = Dispatcher(
    "pre_execute",
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
@pre_execute.register(ops.Node, BaseBackend)
def pre_execute_default(node, *clients, **kwargs):
    return Scope()


# Merge the results of all client pre-execution with scope
@pre_execute.register(ops.Node, [BaseBackend])
def pre_execute_multiple_clients(node, *clients, scope=None, **kwargs):
    scope = scope.merge_scopes(
        list(map(partial(pre_execute, node, scope=scope, **kwargs), clients))
    )
    return scope


execute_literal = Dispatcher(
    "execute_literal",
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
    "post_execute",
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


execute = Dispatcher("execute")
