from __future__ import absolute_import

import collections

import itertools

import six

import numpy as np
import pandas as pd

from pandas.core.groupby import SeriesGroupBy

import toolz

import ibis.expr.datatypes as dt
import ibis.expr.signature as sig
import ibis.expr.operations as ops

from ibis.pandas.core import scalar_types
from ibis.pandas.dispatch import execute_node, Dispatcher, pause_ordering
from ibis.compat import (
    functools, signature, Parameter, _empty as empty, viewkeys)


rule_to_python_type = Dispatcher(
    'rule_to_python_type',
    doc="""\
Convert an ibis :class:`~ibis.expr.datatypes.DataType` into a pandas backend
friendly ``multipledispatch`` signature.

Parameters
----------
rule : DataType
    The :class:`~ibis.expr.datatypes.DataType` subclass to map to a pandas
    friendly type.

Returns
-------
types : Union[type, Tuple[type...]]
    A pandas backend friendly signature
"""
)


def arguments_from_signature(signature, *args, **kwargs):
    """Validate signature against `args` and `kwargs` and return the kwargs
    asked for in the signature

    Parameters
    ----------
    args : Tuple[object...]
    kwargs : Dict[str, object]

    Returns
    -------
    args, kwargs : Tuple[Tuple[object...], Dict[str, object]]

    Examples
    --------
    >>> from ibis.compat import signature
    >>> def foo(a, b=1):
    ...     return a + b
    >>> foo_sig = signature(foo)
    >>> args, kwargs = arguments_from_signature(foo_sig, 1, b=2)
    >>> args
    (1,)
    >>> kwargs
    {'b': 2}
    >>> def bar(a):
    ...     return a + 1
    >>> bar_sig = signature(bar)
    >>> args, kwargs = arguments_from_signature(bar_sig, 1, b=2)
    >>> args
    (1,)
    >>> kwargs
    {}
    """
    bound = signature.bind_partial(*args)
    meta_kwargs = toolz.merge({'kwargs': kwargs}, kwargs)
    remaining_parameters = viewkeys(signature.parameters) - viewkeys(
        bound.arguments
    )
    new_kwargs = {
        k: meta_kwargs[k] for k in remaining_parameters
        if k in signature.parameters
        if signature.parameters[k].kind in {
            Parameter.KEYWORD_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.VAR_KEYWORD,
        }
    }
    return args, new_kwargs


@rule_to_python_type.register(dt.DataType)
def datatype_rule(rule):
    return scalar_types


@rule_to_python_type.register(dt.Array)
def array_rule(rule):
    return (list,)


@rule_to_python_type.register(dt.Map)
def map_rule(rule):
    return (dict,)


@rule_to_python_type.register(dt.Struct)
def struct_rule(rule):
    return (collections.OrderedDict,)


@rule_to_python_type.register(dt.String)
def string_rule(rule):
    return six.string_types


@rule_to_python_type.register(dt.Integer)
def int_rule(rule):
    return six.integer_types + (np.integer,)


@rule_to_python_type.register(dt.Floating)
def float_rule(rule):
    return (float, np.floating)


def nullable(datatype):
    """Return the signature of a scalar value that is allowed to be NULL (in
    SQL parlance).

    Parameters
    ----------
    datatype : ibis.expr.datatypes.DataType

    Returns
    -------
    Tuple[Type]
    """
    return (type(None),) if datatype.nullable else ()


def udf_signature(input_type, klass):
    """Compute the appropriate signature for an ibis Node from a list of input
    types.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of ibis DataType instances
    klass : Type
        pd.Series or SeriesGroupBy

    Returns
    -------
    Tuple[Type]
        A tuple of types appropriate for use in a multiple dispatch signature.

    Examples
    --------
    >>> from pprint import pprint
    >>> import pandas as pd
    >>> from pandas.core.groupby import SeriesGroupBy
    >>> import ibis.expr.datatypes as dt
    >>> input_type = [dt.int64, dt.double]
    >>> sig = udf_signature(input_type, pd.Series)
    >>> pprint(sig)  # doctest: +ELLIPSIS
    ((<class '...Series'>,
      <... 'int'>,
      <... 'numpy.integer'>,
      <... 'NoneType'>),
     (<class '...Series'>,
      <... 'float'>,
      <... 'numpy.floating'>,
      <... 'NoneType'>))
    >>> input_type = [dt.Int64(nullable=False), dt.Double(nullable=False)]
    >>> sig = udf_signature(input_type, SeriesGroupBy)
    >>> pprint(sig)  # doctest: +ELLIPSIS
    ((<class '...SeriesGroupBy'>,
      <... 'int'>,
      <... 'numpy.integer'>),
     (<class '...SeriesGroupBy'>,
      <... 'float'>,
      <... 'numpy.floating'>))
    """
    return tuple(
        (klass,) + rule_to_python_type(r) + nullable(r) for r in input_type
    )


def check_matching_signature(input_type):
    """Make sure that the number of arguments declared by the user in
    `input_type` matches that of the wrapped function's signature.

    Parameters
    ----------
    input_type : List[DataType]

    Returns
    -------
    wrapper : callable
    """
    def wrapper(func):
        num_params = sum(
            param.kind in {param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY}
            for param in signature(func).parameters.values()
            if param.default is empty
        )
        num_declared = len(input_type)
        if num_params != num_declared:
            raise TypeError(
                'Function {!r} has {:d} parameters, '
                'input_type has {:d}. These must match'.format(
                    func.__name__,
                    num_params,
                    num_declared,
                )
            )

        return func
    return wrapper


def udf(input_type, output_type):
    """Define a UDF (user-defined function) that operates element wise on a
    Pandas Series.

    Parameters
    ----------
    input_type : List[ibis.rules]
        A rules encoding the abstract type of each argument. These are found in
        :mod:`~ibis.expr.rules`. The length of this list must match the number
        of arguments to the function. Argument splatting in the UDF signature
        is not supported.
    output_type : ibis.rules or ibis.expr.datatypes.DataType
        The return type of the function.

    Examples
    --------
    >>> import ibis
    >>> @udf(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length(series):
    ...     return series.str.len() * 2
    """
    def wrapper(func):
        # generate a new custom node

        UDFNode = type(
            func.__name__,
            (ops.ValueOp,),
            {
                'signature': sig.TypeSignature.from_dtypes(input_type),
                'output_type': output_type.array_type
            }
        )

        # Don't reorder the multiple dispatch graph for each of these
        # definitions
        with pause_ordering():

            # Define a execution rule for a simple elementwise Series function
            @execute_node.register(
                UDFNode, *udf_signature(input_type, klass=pd.Series)
            )
            def execute_udf_node(op, *args, **kwargs):
                args, kwargs = arguments_from_signature(
                    signature(func), *args, **kwargs
                )
                return func(*args, **kwargs)

            # Define an execution rule for elementwise operations on a grouped
            # Series
            @execute_node.register(
                UDFNode, *udf_signature(input_type, klass=SeriesGroupBy)
            )
            def execute_udf_node_groupby(op, *args, **kwargs):
                groupers = [
                    grouper for grouper in (
                        getattr(arg, 'grouper', None) for arg in args
                    ) if grouper is not None
                ]

                # all grouping keys must be identical
                assert all(groupers[0] == grouper for grouper in groupers[1:])

                # we're performing a scalar operation on grouped column, so
                # perform the operation directly on the underlying Series and
                # regroup after it's finished
                arguments = [getattr(arg, 'obj', arg) for arg in args]
                groupings = groupers[0].groupings
                args, kwargs = arguments_from_signature(
                    signature(func), *arguments, **kwargs
                )
                return func(*args, **kwargs).groupby(groupings)

        @check_matching_signature(input_type)
        @functools.wraps(func)
        def wrapped(*args):
            return UDFNode(*args).to_expr()

        return wrapped

    return wrapper


def udaf(input_type, output_type):
    """Define a UDAF (user-defined aggregation function) that takes N pandas.
    Series or scalar values as inputs.

    Parameters
    ----------
    input_type : List[T]
        A rules encoding the abstract type of each argument or one of the types
        found in :mod:`~ibis.expr.datatypes`. These are found in
        :mod:`~ibis.expr.rules`. The length of this list must match the number
        of arguments to the function. Argument splatting in the UDF signature
        is not supported.
    output_type : rule or ibis.expr.datatypes.DataType
        The abstract return type of the function. This *cannot* be a rule that
        encodes a :class:`~ibis.expr.types.ColumnExpr` since this API is for
        defining *aggregation* functions.

    Examples
    --------
    >>> import ibis
    >>> @udaf(input_type=[dt.string], output_type=dt.int64)
    ... def my_string_length_agg(series, **kwargs):
    ...     return (series.str.len() * 2).sum()
    """
    def wrapper(func):
        UDAFNode = type(
            func.__name__,
            (ops.Reduction,),
            {
                'signature': sig.TypeSignature.from_dtypes(input_type),
                'output_type': output_type.scalar_type
            }
        )

        with pause_ordering():

            # An execution rule for a simple aggregate node
            @execute_node.register(
                UDAFNode, *udf_signature(input_type, klass=pd.Series)
            )
            def execute_udaf_node(op, *args, **kwargs):
                args, kwargs = arguments_from_signature(
                    signature(func), *args, **kwargs
                )
                return func(*args, **kwargs)

            # An execution rule for a grouped aggregation node. This includes
            # aggregates applied over a window.
            @execute_node.register(
                UDAFNode, *udf_signature(input_type, klass=SeriesGroupBy)
            )
            def execute_udaf_node_groupby(op, *args, **kwargs):
                # construct a generator that yields the next group of data for
                # every argument excluding the first (pandas performs the
                # iteration for the first argument) for each argument that is a
                # SeriesGroupBy.
                #
                # If the argument is not a SeriesGroupBy then keep repeating it
                # until all groups are exhausted.
                aggcontext = kwargs.pop('aggcontext', None)
                assert aggcontext is not None, 'aggcontext is None'
                iters = (
                    (data for _, data in arg)
                    if isinstance(arg, SeriesGroupBy)
                    else itertools.repeat(arg) for arg in args[1:]
                )
                funcsig = signature(func)

                def aggregator(first, *rest, **kwargs):
                    # map(next, *rest) gets the inputs for the next group
                    # TODO: might be inefficient to do this on every call
                    args, kwargs = arguments_from_signature(
                        funcsig, first, *map(next, rest), **kwargs
                    )
                    return func(*args, **kwargs)

                result = aggcontext.agg(args[0], aggregator, *iters, **kwargs)
                return result

        @check_matching_signature(input_type)
        @functools.wraps(func)
        def wrapped(*args):
            return UDAFNode(*args).to_expr()

        return wrapped

    return wrapper
