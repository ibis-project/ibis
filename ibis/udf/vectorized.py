"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

import functools
from inspect import Parameter, signature

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.expr.operations import (
    AnalyticVectorizedUDF,
    ElementWiseVectorizedUDF,
    ReductionVectorizedUDF,
)


def parameter_count(funcsig):
    """Get the number of positional-or-keyword or position-only parameters in a
    function signature.

    Parameters
    ----------
    funcsig : inspect.Signature
        A UDF signature

    Returns
    -------
    int
        The number of parameters
    """
    return sum(
        param.kind in {param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY}
        for param in funcsig.parameters.values()
        if param.default is Parameter.empty
    )


def valid_function_signature(input_type, func):
    """Check that the declared number of inputs (the length of `input_type`)
    and the number of inputs to `func` are equal.

    Parameters
    ----------
    input_type : List[DataType]
    func : callable

    Returns
    -------
    inspect.Signature
    """
    funcsig = signature(func)
    declared_parameter_count = len(input_type)
    function_parameter_count = parameter_count(funcsig)

    if declared_parameter_count != function_parameter_count:
        raise TypeError(
            'Function signature {!r} has {:d} parameters, '
            'input_type has {:d}. These must match. Non-column '
            'parameters must be defined as keyword only, i.e., '
            'def foo(col, *, function_param).'.format(
                func.__name__,
                function_parameter_count,
                declared_parameter_count,
            )
        )
    return funcsig


class UserDefinedFunction(object):
    """ Class representing a user defined function.

    This class Implements __call__ that returns an ibis expr for the UDF.
    """

    def __init__(self, func, func_type, input_type, output_type):
        valid_function_signature(input_type, func)

        self.func = func
        self.func_type = func_type

        self.input_type = list(map(dt.dtype, input_type))

        if isinstance(output_type, list):
            try:
                (output_type,) = output_type
            except ValueError:
                raise com.IbisTypeError(
                    'The output type of a UDF must be either a single '
                    'datatype, or equivalently, a single datatype wrapped in '
                    'a list.'
                )

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
    input_type = list(map(dt.dtype, input_type))
    output_type = dt.dtype(output_type)

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

    Define a UDF with non-column parameters:

    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series, *, times):
    ...     return series.str.len() * times
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
    """
    return _udf_decorator(ReductionVectorizedUDF, input_type, output_type)
