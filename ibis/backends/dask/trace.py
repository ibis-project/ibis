"""Module that adds tracing to dask execution.

With tracing enabled, this module will log time and call stack information of
the executed expression. Call stack information is presented with indentation
level.
For example:
import dask.dataframe as dd
import pandas as pd
import logging
import ibis.expr.datatypes as dt
import ibis.dask
from ibis.legacy.udf.vectorized import elementwise
from ibis.backends.dask import trace
logging.basicConfig()
trace.enable()
df = dd.from_pandas(
    pd.DataFrame({'a': [1, 2, 3]})
)
con = ibis.dask.connect({"table1": df})
@elementwise(
    input_type=[dt.double],
    output_type=dt.double
)
def add_one(v):
    import time
    time.sleep(5)
    return v + 1
table = con.table("table1")
table = table.mutate(b=add_one(table['a']))
table.execute()
Output:
DEBUG:ibis.dask.trace: main_execute Selection
DEBUG:ibis.dask.trace:   execute_until_in_scope Selection
DEBUG:ibis.dask.trace:     execute_until_in_scope DaskTable
DEBUG:ibis.dask.trace:       execute_database_table_client DaskTable
DEBUG:ibis.dask.trace:       execute_database_table_client DaskTable 0:00:00.000085
DEBUG:ibis.dask.trace:     execute_until_in_scope DaskTable 0:00:00.000362
DEBUG:ibis.dask.trace:     execute_selection_dataframe Selection
DEBUG:ibis.dask.trace:       main_execute ElementWiseVectorizedUDF
DEBUG:ibis.dask.trace:         execute_until_in_scope ElementWiseVectorizedUDF
DEBUG:ibis.dask.trace:           execute_until_in_scope TableColumn
DEBUG:ibis.dask.trace:             execute_until_in_scope DaskTable
DEBUG:ibis.dask.trace:             execute_until_in_scope DaskTable 0:00:00.000061
DEBUG:ibis.dask.trace:             execute_table_column_df_or_df_groupby TableColumn
DEBUG:ibis.dask.trace:             execute_table_column_df_or_df_groupby TableColumn 0:00:00.000304  # noqa: E501
DEBUG:ibis.dask.trace:           execute_until_in_scope TableColumn 0:00:00.000584
DEBUG:ibis.dask.trace:           execute_udf_node ElementWiseVectorizedUDF
DEBUG:ibis.dask.trace:           execute_udf_node ElementWiseVectorizedUDF 0:00:05.019173
DEBUG:ibis.dask.trace:         execute_until_in_scope ElementWiseVectorizedUDF 0:00:05.052604  # noqa: E501
DEBUG:ibis.dask.trace:       main_execute ElementWiseVectorizedUDF 0:00:05.052819
DEBUG:ibis.dask.trace:     execute_selection_dataframe Selection 0:00:05.054894
DEBUG:ibis.dask.trace:   execute_until_in_scope Selection 0:00:05.055662
DEBUG:ibis.dask.trace: main_execute Selection 0:00:05.056556.
"""

from __future__ import annotations

import functools
import logging
import traceback
from datetime import datetime

import ibis
from ibis.backends.pandas.dispatcher import TwoLevelDispatcher
from ibis.config import options
from ibis.expr import types as ir

_logger = logging.getLogger("ibis.dask.trace")

# A list of funcs that is traced
_trace_funcs = set()


def enable():
    """Enable tracing."""
    if options.dask is None:
        # dask options haven't been registered yet - force module __getattr__
        ibis.dask  # noqa: B018

    options.dask.enable_trace = True
    logging.getLogger("ibis.dask.trace").setLevel(logging.DEBUG)


def _log_trace(func, start=None):
    level = 0
    current_frame = None

    # Increase the current level for each traced function in the stackframe
    # This way we can visualize the call stack.
    for frame, _ in traceback.walk_stack(None):
        current_frame = current_frame if current_frame is not None else frame
        func_name = frame.f_code.co_name
        if func_name in _trace_funcs:
            level += 1

    # We can assume we have 'args' because we only call _log_trace inside
    # trace or TraceDispatcher.register
    current_op = current_frame.f_locals["args"][0]

    # If the first argument is a Expr, we print its op because it's more
    # informative.
    if isinstance(current_op, ir.Expr):
        current_op = current_op.op()

    _logger.debug(
        "%s %s %s %s",
        "  " * level,
        func.__name__,
        type(current_op).__qualname__,
        f"{datetime.now() - start}" if start else "",
    )


def trace(func):
    """Return a function decorator that wraps `func` with tracing."""
    _trace_funcs.add(func.__name__)

    @functools.wraps(func)
    def traced_func(*args, **kwargs):
        import ibis

        # Similar to the pandas backend, it is possible to call this function
        # without having initialized the configuration option. This can happen
        # when tests are distributed across multiple processes, for example.
        ibis.dask  # noqa: B018

        if not options.dask.enable_trace:
            return func(*args, **kwargs)
        else:
            start = datetime.now()
            _log_trace(func)
            res = func(*args, **kwargs)
            _log_trace(func, start)
            return res

    return traced_func


class TraceTwoLevelDispatcher(TwoLevelDispatcher):
    """A Dispatcher that also wraps the registered function with tracing."""

    def __init__(self, name, doc=None):
        super().__init__(name, doc)

    def register(self, *types, **kwargs):
        """Register a function with this Dispatcher.

        The function will also be wrapped with tracing information.
        """

        def _(func):
            trace_func = trace(func)
            TwoLevelDispatcher.register(self, *types, **kwargs)(trace_func)
            # return func instead trace_func here so that
            # chained register didn't get wrapped multiple
            # times
            return func

        return _
