"""Validation for UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""

from inspect import Parameter, _ParameterKind, signature
from typing import Any, Callable, List, Set

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


def _parameter_kinds(funcsig: signature) -> Set[_ParameterKind]:
    """Returns a set containing the kinds of parameters that are defined in
    the function signature. For example, a function with **kwargs would have
    typing._ParameterKind.VAR_KEYWORD in its set of parameter kinds.

    Parameters
    ----------
    funcsig : inspect.Signature
        A UDF signature

    Returns
    -------
    Set[typing._ParameterKind]
        Set containing the kinds of parameters in the function signature
    """
    return {param.kind for param in funcsig.parameters.values()}


def validate_parameter_kinds(func: Callable) -> None:
    """Check that the kinds of parameters used in the signature of `func` is
    valid, and raises an error otherwise.

    The signature is invalid if both positional-only arguments and *args are
    used at the same time in the function signature.

    Parameters
    ----------
    func : callable
    """
    kinds = _parameter_kinds(signature(func))

    if Parameter.VAR_POSITIONAL in kinds and (
        Parameter.POSITIONAL_ONLY in kinds
        or Parameter.POSITIONAL_OR_KEYWORD in kinds
    ):
        raise com.IbisError(
            'UDF signature cannot have both positional arguments and *args'
        )


def validate_input_type(
    input_type: List[DataType], func: Callable
) -> signature:
    """Check that the declared number of inputs (the length of `input_type`)
    and the number of inputs to `func` are equal.

    If `func` is defined to use *args rather than a fixed number of positional
    arguments, then no check is done (since no check can be done).

    Parameters
    ----------
    input_type : List[DataType]
    func : callable

    Returns
    -------
    inspect.Signature
    """
    funcsig = signature(func)

    if Parameter.VAR_POSITIONAL not in _parameter_kinds(funcsig):
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
