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

# Given a node, compute a (possibly partial) scope prior to regular execution
# This is useful if parts of the tree structure need to be executed at the
# same time or if there are other reasons to need to interrupt the
# regular depth-first traversal of the tree
pre_execute = Dispatcher('pre_execute')


# Default does nothing
@data_preload.register(object, object)
def data_preload_default(node, data, **kwargs):
    return data


# Default returns an empty scope
@pre_execute.register(object)
def pre_execute_default(node, **kwargs):
    return {}
