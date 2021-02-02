from multipledispatch import Dispatcher

import ibis.backends.pandas.core as core_dispatch
import ibis.backends.pandas.dispatch as pandas_dispatch
from ibis.backends.dask.trace import TraceTwoLevelDispatcher

dask_execute_node = TraceTwoLevelDispatcher('dask_execute_node')
for types, func in pandas_dispatch.execute_node.funcs.items():
    dask_execute_node.register(*types)(func)

dask_execute = Dispatcher('dask_execute')
dask_execute.funcs.update(core_dispatch.execute.funcs)

dask_pre_execute = Dispatcher('dask_pre_execute')
dask_pre_execute.funcs.update(core_dispatch.pre_execute.funcs)

dask_execute_literal = Dispatcher('dask_execute_literal')
dask_execute_literal.funcs.update(core_dispatch.execute_literal.funcs)

dask_post_execute = Dispatcher('dask_post_execute')
dask_post_execute.funcs.update(core_dispatch.post_execute.funcs)
