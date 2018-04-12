from __future__ import absolute_import

import contextlib

import ibis.common as com
import ibis.expr.operations as ops
from multipledispatch import Dispatcher, halt_ordering, restart_ordering


execute = Dispatcher('execute', doc='Execute an expression')

# Individual operation execution
execute_node = Dispatcher(
    'execute_node',
    doc=(
        'Execute an individual operation given the operation and its computed '
        'arguments'
    )
)


@execute_node.register(ops.Node)
def execute_node_without_scope(node, **kwargs):
    raise com.UnboundExpressionError(
        'Node of type {!r} has no data bound to it. '
        'You probably tried to execute an expression without a data source.'
    )


execute_first = Dispatcher(
    'execute_first', doc='Compute from the top of the expression downward')

data_preload = Dispatcher(
    'data_preload', doc='Possibly preload data from the client, given a node')

pre_execute = Dispatcher(
    'pre_execute',
    doc="""\
Given a node, compute a (possibly partial) scope prior to standard execution.

Notes
-----
This function is useful if parts of the tree structure need to be executed at
the same time or if there are other reasons to need to interrupt the regular
depth-first traversal of the tree.
""")


# Default does nothing
@data_preload.register(object, object)
def data_preload_default(node, data, **kwargs):
    return data


# Default returns an empty scope
@pre_execute.register(object, object)
def pre_execute_default(node, client, **kwargs):
    return {}


@contextlib.contextmanager
def pause_ordering():
    """Pause multipledispatch ordering."""
    halt_ordering()
    try:
        yield
    finally:
        restart_ordering()
