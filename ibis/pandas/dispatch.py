from __future__ import absolute_import

from multipledispatch import Dispatcher


execute = Dispatcher('execute')
execute_node = Dispatcher('execute_node')
