from __future__ import absolute_import

from multipledispatch import Dispatcher


# Main interface to execution; ties the following functions together
execute = Dispatcher('execute')

# Individual operation execution
execute_node = Dispatcher('execute_node')

# Compute from the top of the expression downward
execute_first = Dispatcher('execute_first')

# Possibly preload data from the client, given a node
data_preload = Dispatcher('data_preload')


# Default does nothing
@data_preload.register(object, object)
def data_preload_default(node, data, **kwargs):
    return data
