"""Validation for UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

from inspect import Parameter, signature
from typing import Any, Callable, List

import ibis.common.exceptions as com
from ibis.expr.datatypes import DataType


def _parameter_count(funcsig: signature) -> int:
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


def validate_input_type(
    input_type: List[DataType], func: Callable
) -> signature:
    """Check that the declared number of inputs (the length of `input_type`)
    and the number of inputs to `func` are equal.

    If the signature of `func` uses *args, then no check is done (since no
    check can be done).

    Parameters
    ----------
    input_type : List[DataType]
    func : callable

    Returns
    -------
    inspect.Signature
    """
    funcsig = signature(func)
    params = funcsig.parameters.values()

    # We can only do validation if all the positional arguments are explicit
    # (i.e. no *args)
    if not any([param.kind is Parameter.VAR_POSITIONAL for param in params]):
        declared_parameter_count = len(input_type)
        function_parameter_count = _parameter_count(funcsig)

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


def validate_output_type(output_type: Any) -> None:
    """Check that the output type is a single datatype."""

    if isinstance(output_type, list):
        raise com.IbisTypeError(
            'The output type of a UDF must be a single datatype.'
        )
