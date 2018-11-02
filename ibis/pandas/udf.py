"""APIs for creating user-defined element-wise, reduction and analytic
functions.
"""

from __future__ import absolute_import

import collections
import functools
import itertools
import operator

from inspect import Parameter, signature

import numpy as np
import pandas as pd

from pandas.core.groupby import SeriesGroupBy

import toolz

from multipledispatch import Dispatcher

import ibis.expr.datatypes as dt
import ibis.expr.signature as sig
import ibis.expr.operations as ops

from ibis.pandas.core import scalar_types
from ibis.pandas.dispatch import execute_node


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
Union[Type[U], Tuple[Type[T], ...]]
    A pandas-backend-friendly signature
""")


def arguments_from_signature(signature, *args, **kwargs):
    """Validate signature against `args` and `kwargs` and return the kwargs
    asked for in the signature

    Parameters
    ----------
    args : Tuple[object...]
    kwargs : Dict[str, object]

    Returns
    -------
    Tuple[Tuple, Dict[str, Any]]

    Examples
    --------
    >>> from inspect import signature
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
    remaining_parameters = signature.parameters.keys() - bound.arguments.keys()
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
    return str


@rule_to_python_type.register(dt.Integer)
def int_rule(rule):
    return int + (np.integer,)


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


def udf_signature(input_type, pin, klass):
    """Compute the appropriate signature for a
    :class:`~ibis.expr.operations.Node` from a list of input types
    `input_type`.

    Parameters
    ----------
    input_type : List[ibis.expr.datatypes.DataType]
        A list of :class:`~ibis.expr.datatypes.DataType` instances representing
        the signature of a UDF/UDAF.
    pin : Optional[int]
        If this is not None, pin the `pin`-th argument type to `klass`
    klass : Union[Type[pd.Series], Type[SeriesGroupBy]]
        The pandas object that every argument type should contain

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
    >>> input_type = [dt.string, dt.double]
    >>> sig = udf_signature(input_type, pin=None, klass=pd.Series)
    >>> pprint(sig)  # doctest: +ELLIPSIS
    ((<class '...Series'>, <... '...str...'>, <... 'NoneType'>),
     (<class '...Series'>,
      <... 'float'>,
      <... 'numpy.floating'>,
      <... 'NoneType'>))
    >>> not_nullable_types = [
    ...     dt.String(nullable=False), dt.Double(nullable=False)]
    >>> sig = udf_signature(not_nullable_types, pin=None, klass=pd.Series)
    >>> pprint(sig)  # doctest: +ELLIPSIS
    ((<class '...Series'>, <... '...str...'>),
     (<class '...Series'>,
      <... 'float'>,
      <... 'numpy.floating'>))
    >>> sig0 = udf_signature(input_type, pin=0, klass=SeriesGroupBy)
    >>> sig1 = udf_signature(input_type, pin=1, klass=SeriesGroupBy)
    >>> pprint(sig0)  # doctest: +ELLIPSIS
    (<class '...SeriesGroupBy'>,
     (<class '...SeriesGroupBy'>,
      <... 'float'>,
      <... 'numpy.floating'>,
      <... 'NoneType'>))
    >>> pprint(sig1)  # doctest: +ELLIPSIS
    ((<class '...SeriesGroupBy'>,
      <... '...str...'>,
      <... 'NoneType'>),
     <class '...SeriesGroupBy'>)
    """
    nargs = len(input_type)

    if not nargs:
        return ()

    if nargs == 1:
        r, = input_type
        result = (klass,) + rule_to_python_type(r) + nullable(r)
        return (result,)

    return tuple(
        klass if pin is not None and pin == i else
        ((klass,) + rule_to_python_type(r) + nullable(r))
        for i, r in enumerate(input_type)
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


class udf:
    @staticmethod
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
        >>> from ibis.pandas.udf import udf
        >>> @udf.elementwise(input_type=[dt.string], output_type=dt.int64)
        ... def my_string_length(series):
        ...     return series.str.len() * 2
        """
        def wrapper(func):
            # validate that the input_type argument and the function signature
            # match
            funcsig = valid_function_signature(input_type, func)

            # generate a new custom node
            UDFNode = type(
                func.__name__,
                (ops.ValueOp,),
                {
                    'signature': sig.TypeSignature.from_dtypes(input_type),
                    'output_type': output_type.array_type
                }
            )

            # definitions
            # Define an execution rule for a simple elementwise Series
            # function
            @execute_node.register(
                UDFNode,
                *udf_signature(input_type, pin=None, klass=pd.Series))
            @execute_node.register(
                UDFNode,
                *(rule_to_python_type(argtype) + nullable(argtype)
                    for argtype in input_type))
            def execute_udf_node(op, *args, **kwargs):
                args, kwargs = arguments_from_signature(
                    funcsig, *args, **kwargs
                )
                return func(*args, **kwargs)

            # Define an execution rule for elementwise operations on a
            # grouped Series
            nargs = len(input_type)
            group_by_signatures = [
                udf_signature(input_type, pin=pin, klass=SeriesGroupBy)
                for pin in range(nargs)
            ]

            @toolz.compose(*(execute_node.register(UDFNode, *types)
                             for types in group_by_signatures))
            def execute_udf_node_groupby(op, *args, **kwargs):
                groupers = [
                    grouper for grouper in (
                        getattr(arg, 'grouper', None) for arg in args
                    ) if grouper is not None
                ]

                # all grouping keys must be identical
                assert all(
                    groupers[0] == grouper for grouper in groupers[1:])

                # we're performing a scalar operation on grouped column, so
                # perform the operation directly on the underlying Series
                # and regroup after it's finished
                arguments = [getattr(arg, 'obj', arg) for arg in args]
                groupings = groupers[0].groupings
                args, kwargs = arguments_from_signature(
                    signature(func), *arguments, **kwargs
                )
                return func(*args, **kwargs).groupby(groupings)

            @functools.wraps(func)
            def wrapped(*args):
                return UDFNode(*args).to_expr()
            return wrapped
        return wrapper

    @staticmethod
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
        >>> from ibis.pandas.udf import udf
        >>> @udf.reduction(input_type=[dt.string], output_type=dt.int64)
        ... def my_string_length_agg(series, **kwargs):
        ...     return (series.str.len() * 2).sum()
        """
        return udf._grouped(
            input_type, output_type,
            base_class=ops.Reduction,
            output_type_method=operator.attrgetter('scalar_type'))

    @staticmethod
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
        >>> from ibis.pandas.udf import udf
        >>> @udf.analytic(input_type=[dt.double], output_type=dt.double)
        ... def zscore(series):  # note the use of aggregate functions
        ...     return (series - series.mean()) / series.std()
        """
        return udf._grouped(
            input_type, output_type,
            base_class=ops.AnalyticOp,
            output_type_method=operator.attrgetter('array_type'))

    @staticmethod
    def _grouped(input_type, output_type, base_class,
                 output_type_method):
        """Define a user-defined function that is applied per group.

        Parameters
        ----------
        input_type : List[ibis.expr.datatypes.DataType]
            A list of the types found in :mod:`~ibis.expr.datatypes`. The
            length of this list must match the number of arguments to the
            function. Variadic arguments are not yet supported.
        output_type : ibis.expr.datatypes.DataType
            The return type of the function.
        base_class : Type[T]
            The base class of the generated Node
        output_type_method : Callable
            A callable that determines the method to call to get the expression
            type of the UDF

        See Also
        --------
        ibis.pandas.udf.reduction
        ibis.pandas.udf.analytic
        """
        def wrapper(func):
            funcsig = valid_function_signature(input_type, func)

            UDAFNode = type(
                func.__name__,
                (base_class,),
                {
                    'signature': sig.TypeSignature.from_dtypes(input_type),
                    'output_type': output_type_method(output_type),
                }
            )

            # An execution rule for a simple aggregate node
            @execute_node.register(
                UDAFNode,
                *udf_signature(input_type, pin=None, klass=pd.Series))
            def execute_udaf_node(op, *args, **kwargs):
                args, kwargs = arguments_from_signature(
                    funcsig, *args, **kwargs
                )
                return func(*args, **kwargs)

            # An execution rule for a grouped aggregation node. This
            # includes aggregates applied over a window.
            nargs = len(input_type)
            group_by_signatures = [
                udf_signature(input_type, pin=pin, klass=SeriesGroupBy)
                for pin in range(nargs)
            ]

            @toolz.compose(*(execute_node.register(UDAFNode, *types)
                             for types in group_by_signatures))
            def execute_udaf_node_groupby(op, *args, **kwargs):
                # construct a generator that yields the next group of data
                # for every argument excluding the first (pandas performs
                # the iteration for the first argument) for each argument
                # that is a SeriesGroupBy.
                #
                # If the argument is not a SeriesGroupBy then keep
                # repeating it until all groups are exhausted.
                aggcontext = kwargs.pop('aggcontext', None)
                assert aggcontext is not None, 'aggcontext is None'
                iters = (
                    (data for _, data in arg)
                    if isinstance(arg, SeriesGroupBy)
                    else itertools.repeat(arg)
                    for arg in args[1:]
                )
                funcsig = signature(func)

                def aggregator(first, *rest, **kwargs):
                    # map(next, *rest) gets the inputs for the next group
                    # TODO: might be inefficient to do this on every call
                    args, kwargs = arguments_from_signature(
                        funcsig, first, *map(next, rest), **kwargs
                    )
                    return func(*args, **kwargs)

                result = aggcontext.agg(
                    args[0], aggregator, *iters, **kwargs)
                return result

            @functools.wraps(func)
            def wrapped(*args):
                return UDAFNode(*args).to_expr()
            return wrapped
        return wrapper
