import functools
import logging
import traceback
from datetime import datetime

from ibis.config import get_option, set_option
from ibis.expr import types as ir

from .dispatcher import TwoLevelDispatcher

"""Module that adds tracing to pandas execution.

With tracing enabled, this module will log time and call stack information of
the executed expression. Call stack information is presented with indentation
level.

For example:

import pandas as pd
import logging

import ibis.expr.datatypes as dt
import ibis.backends.pandas
from ibis.udf.vectorized import elementwise
from ibis.backends.pandas import trace

logging.basicConfig()
trace.enable()

df = pd.DataFrame(
    {
        'a': [1, 2, 3]
    }
)

con = ibis.backends.pandas.connect({"table1": df})

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

DEBUG:ibis.backends.pandas.trace: main_execute Selection
DEBUG:ibis.backends.pandas.trace:   execute_until_in_scope Selection
DEBUG:ibis.backends.pandas.trace:     execute_until_in_scope PandasTable
DEBUG:ibis.backends.pandas.trace:       execute_database_table_client PandasTable
DEBUG:ibis.backends.pandas.trace:       execute_database_table_client PandasTable 0:00:00.000085
DEBUG:ibis.backends.pandas.trace:     execute_until_in_scope PandasTable 0:00:00.000362
DEBUG:ibis.backends.pandas.trace:     execute_selection_dataframe Selection
DEBUG:ibis.backends.pandas.trace:       main_execute ElementWiseVectorizedUDF
DEBUG:ibis.backends.pandas.trace:         execute_until_in_scope ElementWiseVectorizedUDF
DEBUG:ibis.backends.pandas.trace:           execute_until_in_scope TableColumn
DEBUG:ibis.backends.pandas.trace:             execute_until_in_scope PandasTable
DEBUG:ibis.backends.pandas.trace:             execute_until_in_scope PandasTable 0:00:00.000061
DEBUG:ibis.backends.pandas.trace:             execute_table_column_df_or_df_groupby TableColumn
DEBUG:ibis.backends.pandas.trace:             execute_table_column_df_or_df_groupby TableColumn 0:00:00.000304  # noqa: E501
DEBUG:ibis.backends.pandas.trace:           execute_until_in_scope TableColumn 0:00:00.000584
DEBUG:ibis.backends.pandas.trace:           execute_udf_node ElementWiseVectorizedUDF
DEBUG:ibis.backends.pandas.trace:           execute_udf_node ElementWiseVectorizedUDF 0:00:05.019173
DEBUG:ibis.backends.pandas.trace:         execute_until_in_scope ElementWiseVectorizedUDF 0:00:05.052604  # noqa: E501
DEBUG:ibis.backends.pandas.trace:       main_execute ElementWiseVectorizedUDF 0:00:05.052819
DEBUG:ibis.backends.pandas.trace:     execute_selection_dataframe Selection 0:00:05.054894
DEBUG:ibis.backends.pandas.trace:   execute_until_in_scope Selection 0:00:05.055662
DEBUG:ibis.backends.pandas.trace: main_execute Selection 0:00:05.056556

"""

_logger = logging.getLogger('ibis.backends.pandas.trace')

# A list of funcs that is traced
_trace_funcs = set()
_trace_root = "main_execute"
_TRACE_CONFIG = 'pandas.enable_trace'


def enable():
    """Enable tracing."""
    set_option(_TRACE_CONFIG, True)
    logging.getLogger('ibis.backends.pandas.trace').setLevel(logging.DEBUG)


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
    # trace or TraceDispatcher.resgister
    current_op = current_frame.f_locals['args'][0]

    # If the first argument is a Expr, we print its op because it's more
    # informative.
    if isinstance(current_op, ir.Expr):
        current_op = current_op.op()

    _logger.debug(
        f"{'  ' * level} {func.__name__} {type(current_op).__qualname__} "
        f"{datetime.now() - start if start else ''}"
    )


def trace(func):
    """ Return a function decorator that wraped the decorated function with
    tracing.
    """
    _trace_funcs.add(func.__name__)

    @functools.wraps(func)
    def traced_func(*args, **kwargs):
        trace_enabled = get_option(_TRACE_CONFIG)

        if not trace_enabled:
            return func(*args, **kwargs)
        else:
            start = datetime.now()
            _log_trace(func)
            res = func(*args, **kwargs)
            _log_trace(func, start)
            return res

    return traced_func


class TraceTwoLevelDispatcher(TwoLevelDispatcher):
    """ A Dispatcher that also wraps the registered function with tracing."""

    def __init__(self, name, doc=None):
        super().__init__(name, doc)

    def register(self, *types, **kwargs):
        """ Register a function with this Dispatcher.

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
