from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.operations as ops

from .trace import TraceTwoLevelDispatcher

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


execute = Dispatcher("execute")
