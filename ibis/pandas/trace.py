import functools
import logging
import traceback
from datetime import datetime

from multipledispatch import Dispatcher

from ibis.expr import types as ir

_logger = logging.getLogger('ibis.pandas.trace')

"""Module that adds tracing to pandas execution.

With tracing enabled, this module will log time and call stack information of
the executed expression. Call stack information is presented with indentation
level.

Example:

import pandas as pd
import ibis.expr.datatypes as dt
import ibis.pandas
from ibis.udf.vectorized import elementwise

import logging

logging.basicConfig(level=logging.DEBUG)

df = pd.DataFrame(
    {
        'a': [1, 2, 3]
    }
)

con = ibis.pandas.connect({"table1": df})

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

DEBUG:root: main_execute Selection
DEBUG:root:   execute_until_in_scope Selection
DEBUG:root:     execute_until_in_scope PandasTable
DEBUG:root:       execute_database_table_client PandasTable
DEBUG:root:       execute_database_table_client PandasTable 0:00:00.000067
DEBUG:root:     execute_until_in_scope PandasTable 0:00:00.000374
DEBUG:root:     execute_selection_dataframe Selection
DEBUG:root:       main_execute ElementWiseVectorizedUDF
DEBUG:root:         execute_until_in_scope ElementWiseVectorizedUDF
DEBUG:root:           execute_until_in_scope TableColumn
DEBUG:root:             execute_until_in_scope PandasTable
DEBUG:root:             execute_until_in_scope PandasTable 0:00:00.000073
DEBUG:root:             execute_table_column_df_or_df_groupby TableColumn
DEBUG:root:             execute_table_column_df_or_df_groupby TableColumn 0:00:00.003948  # noqa: E501
DEBUG:root:           execute_until_in_scope TableColumn 0:00:00.004322
DEBUG:root:           execute_udf_node ElementWiseVectorizedUDF
DEBUG:root:           execute_udf_node ElementWiseVectorizedUDF 0:00:05.031799
DEBUG:root:         execute_until_in_scope ElementWiseVectorizedUDF 0:00:05.036855  # noqa: E501
DEBUG:root:       main_execute ElementWiseVectorizedUDF 0:00:05.037206
DEBUG:root:     execute_selection_dataframe Selection 0:00:05.043668
DEBUG:root:   execute_until_in_scope Selection 0:00:05.044942
DEBUG:root: main_execute Selection 0:00:05.046008

"""

# A list of funcs that is traced
_trace_funcs = set()
_trace_root = "main_execute"


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
    def trace_func(*args, **kwargs):
        start = datetime.now()
        _log_trace(func)
        res = func(*args, **kwargs)
        _log_trace(func, start)
        return res

    return trace_func


class TraceDispatcher(Dispatcher):
    """ A Dispatcher that also wraps the registered function with tracing."""

    def __init__(self, name, doc=None):
        super().__init__(name, doc)
        _trace_funcs.add(name)

    def register(self, *types, **kwargs):
        """ Register a function with this Dispatcher.

        The function will also be wrapped with tracing information.
        """

        def _df(func):
            _trace_funcs.add(func.__name__)

            @functools.wraps(func)
            def trace_func(*args, **kwargs):
                start = datetime.now()
                _log_trace(func)
                res = func(*args, **kwargs)
                _log_trace(func, start)
                return res

            self.add(types, trace_func, **kwargs)
            # return func instead trace_func here so that
            # chained register didn't get wrapped multiple
            # times
            return func

        return _df
