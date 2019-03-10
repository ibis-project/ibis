"""
Dispatchers needed to execute the execution tree nodes.
"""
from multipledispatch import Dispatcher

# Individual operation execution
execute_node = Dispatcher(
    'execute_node',
    doc=(
        'Execute an individual operation given the operation and its computed '
        'arguments'
    )
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
""")
