from __future__ import absolute_import

from functools import partial, singledispatch

import toolz
from multipledispatch import Dispatcher

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.window as win
from ibis.pandas.trace import TraceTwoLevelDispatcher


@singledispatch
def is_computable_input(arg):
    """All inputs are not computable without a specific override."""
    return False


@is_computable_input.register(ibis.client.Client)
@is_computable_input.register(ir.Expr)
@is_computable_input.register(dt.DataType)
@is_computable_input.register(type(None))
@is_computable_input.register(win.Window)
@is_computable_input.register(tuple)
def is_computable_input_arg(arg):
    """Return whether `arg` is a valid computable argument."""
    return True


# Individual operation execution
execute_node = TraceTwoLevelDispatcher(
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
def pre_execute_multiple_clients(
    node, *clients, scope=None, state=None, **kwargs
):
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


execute = Dispatcher("execute")

compute_local_context = Dispatcher(
    'compute_local_context',
    doc="""\

Compute local_context for a node in execution

Notes
-----
For a given node, return with a list of localcontext that are going to be
passed to its children nodes.
local_context is useful when data is not uniquely defined by op tree. e.g.
a TableExpr can represent the query select count(a) from table, but the
result of that is different with time context (20190101, 20200101) vs
(20200101, 20210101), because what data is in "table" also depends on the
time context. And such context may not be global for all nodes. Each node
may have its own context. compuate_local_context computes attributes that
are going to be used in executeion and passes these attributes to children
nodes.
""",
)


@compute_local_context.register(ops.Node)
@compute_local_context.register(ops.Node, ibis.client.Client)
def compute_local_context_default(node, *clients, localcontext=None, **kwargs):
    return [localcontext for arg in node.inputs if is_computable_input(arg)]
