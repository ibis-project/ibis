"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

import functools

import ibis.expr.datatypes as dt
import ibis.udf.validate as v
from ibis.expr.operations import (
    AnalyticVectorizedUDF,
    ElementWiseVectorizedUDF,
    ReductionVectorizedUDF,
)


class UserDefinedFunction(object):
    """ Class representing a user defined function.

    This class Implements __call__ that returns an ibis expr for the UDF.
    """

    def __init__(self, func, func_type, input_type, output_type):
        v.validate_input_type(input_type, func)
        v.validate_output_type(output_type)

        self.func = func
        self.func_type = func_type
        self.input_type = list(map(dt.dtype, input_type))
        self.output_type = dt.dtype(output_type)

    def __call__(self, *args, **kwargs):
        # kwargs cannot be part of the node object because it can contain
        # unhashable object, e.g., list.
        # Here, we keep the node hashable by creating a closure that contains
        # kwargs.
        @functools.wraps(self.func)
        def func(*args):
            return self.func(*args, **kwargs)

        op = self.func_type(
            func=func,
            args=args,
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return op.to_expr()


def _udf_decorator(node_type, input_type, output_type):
    def wrapper(func):
        return UserDefinedFunction(func, node_type, input_type, output_type)

    return wrapper


def analytic(input_type, output_type):
    """Define an *analytic* user-defined function that takes N
    pandas Series or scalar values as inputs and produces N rows of output.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.udf.vectorized import analytic
    >>> @analytic(input_type=[dt.double], output_type=dt.double)
    ... def zscore(series):  # note the use of aggregate functions
    ...     return (series - series.mean()) / series.std()

    Define and use an UDF with multiple return columns:

    >>> @analytic(
    ...     input_type=[dt.double],
    ...     output_type=dt.Struct(['demean', 'zscore'], [dt.double, dt.double])
    ... )
    ... def demean_and_zscore(v):
    ...     mean = v.mean()
    ...     std = v.std()
    ...     return v - mean, (v - mean) / std
    >>>
    >>> win = ibis.window(preceding=None, following=None, group_by='key')
    >>> # add two columns "demean" and "zscore"
    >>> table = table.mutate(
    ...     demean_and_zscore(table['v']).over(win).destructure()
    ... )
    """
    return _udf_decorator(AnalyticVectorizedUDF, input_type, output_type)


def elementwise(input_type, output_type):
    """Define a UDF (user-defined function) that operates element wise on a
    Pandas Series.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.udf.vectorized import elementwise
    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series):
    ...     return series.str.len() * 2

    Define an UDF with non-column parameters:

    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series, *, times):
    ...     return series.str.len() * times

    Define and use an UDF with multiple return columns:

    >>> @elementwise(
    ...     input_type=[dt.string],
    ...     output_type=dt.Struct(['year', 'monthday'], [dt.string, dt.string])
    ... )
    ... def year_monthday(date):
    ...     return date.str.slice(0, 4), date.str.slice(4, 8)
    >>>
    >>> # add two columns "year" and "monthday"
    >>> table = table.mutate(year_monthday(table['date']).destructure())
    """
    return _udf_decorator(ElementWiseVectorizedUDF, input_type, output_type)


def reduction(input_type, output_type):
    """Define a user-defined reduction function that takes N pandas Series
    or scalar values as inputs and produces one row of output.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of the types found in :mod:`~ibis.expr.datatypes`. The
        length of this list must match the number of arguments to the
        function. Variadic arguments are not yet supported.
    output_type : ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> from ibis.udf.vectorized import reduction
    >>> @reduction(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length_agg(series, **kwargs):
    ...     return (series.str.len() * 2).sum()

    Define and use an UDF with multiple return columns:

    >>> @reduction(
    ...     input_type=[dt.double],
    ...     output_type=dt.Struct(['mean', 'std'], [dt.double, dt.double])
    ... )
    ... def mean_and_std(v):
    ...     return v.mean(), v.std()
    >>>
    >>> # create aggregation columns "mean" and "std"
    >>> table = table.groupby('key').aggregate(
    ...     mean_and_std(table['v']).destructure()
    ... )
    """
    return _udf_decorator(ReductionVectorizedUDF, input_type, output_type)
