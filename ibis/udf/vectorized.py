"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

import functools
from inspect import Parameter, signature

import ibis.expr.datatypes as dt
from ibis.expr.operations import (
    AnalyticsVectorizedUDF,
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
            'input_type has {:d}. These must match'.format(
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
        self.input_type = input_type
        self.output_type = output_type

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


def analytics(input_type, output_type):
    """ Reduction vectorized UDFs."""
    return _udf_decorator(AnalyticsVectorizedUDF, input_type, output_type)


def elementwise(input_type, output_type):
    """ Element wise vectorized UDFs."""
    return _udf_decorator(ElementWiseVectorizedUDF, input_type, output_type)


def reduction(input_type, output_type):
    """ Reduction vectorized UDFs."""
    return _udf_decorator(ReductionVectorizedUDF, input_type, output_type)
