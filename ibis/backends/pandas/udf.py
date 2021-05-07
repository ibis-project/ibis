"""APIs for creating user-defined element-wise, reduction and analytic
functions.
"""

from __future__ import absolute_import

import collections
import functools
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.client
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.udf.vectorized
from ibis.util import coerce_to_dataframe

from .aggcontext import Summarize, Transform
from .core import date_types, time_types, timedelta_types, timestamp_types
from .dispatch import execute_node, pre_execute


@functools.singledispatch
def rule_to_python_type(datatype):
    """Convert an ibis :class:`~ibis.expr.datatypes.DataType` into a pandas
    backend friendly ``multipledispatch`` signature.

    Parameters
    ----------
    rule : DataType
        The :class:`~ibis.expr.datatypes.DataType` subclass to map to a pandas
        friendly type.

    Returns
    -------
    Union[Type[U], Tuple[Type[T], ...]]
        A pandas-backend-friendly signature
    """
    raise NotImplementedError(
        "Don't know how to convert type {} into a native Python type".format(
            type(datatype)
        )
    )


def create_gens_from_args_groupby(args: Tuple[SeriesGroupBy]):
    """ Create generators for each args for groupby udaf.

    Returns a generator that outputs each group.

    Parameters
    ----------
    args : Tuple[SeriesGroupBy...]

    Returns
    -------
    Tuple[Generator]
    """
    iters = ((data for _, data in arg) for arg in args)
    return iters


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
    return (str,)


@rule_to_python_type.register(dt.Integer)
def int_rule(rule):
    return int, np.integer


@rule_to_python_type.register(dt.Floating)
def float_rule(rule):
    return float, np.floating


@rule_to_python_type.register(dt.Boolean)
def bool_rule(rule):
    return bool, np.bool_


@rule_to_python_type.register(dt.Interval)
def interval_rule(rule):
    return timedelta_types


@rule_to_python_type.register(dt.Date)
def date_rule(rule):
    return date_types


@rule_to_python_type.register(dt.Timestamp)
def timestamp_rule(rule):
    return timestamp_types


@rule_to_python_type.register(dt.Time)
def time_rule(rule):
    return time_types


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


class udf:
    @staticmethod
    def elementwise(input_type, output_type):
        """Alias for ibis.udf.vectorized.elementwise."""

        return ibis.udf.vectorized.elementwise(input_type, output_type)

    @staticmethod
    def reduction(input_type, output_type):
        """Alias for ibis.udf.vectorized.reduction."""
        return ibis.udf.vectorized.reduction(input_type, output_type)

    @staticmethod
    def analytic(input_type, output_type):
        """Alias for ibis.udf.vectorized.analytic."""
        return ibis.udf.vectorized.analytic(input_type, output_type)


@pre_execute.register(ops.ElementWiseVectorizedUDF)
@pre_execute.register(ops.ElementWiseVectorizedUDF, ibis.client.Client)
def pre_execute_elementwise_udf(op, *clients, scope=None, **kwargs):
    """Register execution rules for elementwise UDFs.
    """
    input_type = op.input_type

    # definitions

    # Define an execution rule for elementwise operations on a
    # grouped Series
    nargs = len(input_type)

    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(SeriesGroupBy, nargs))
    )
    def execute_udf_node_groupby(op, *args, **kwargs):
        func = op.func

        groupers = [
            grouper
            for grouper in (getattr(arg, 'grouper', None) for arg in args)
            if grouper is not None
        ]

        # all grouping keys must be identical
        assert all(groupers[0] == grouper for grouper in groupers[1:])

        # we're performing a scalar operation on grouped column, so
        # perform the operation directly on the underlying Series
        # and regroup after it's finished
        args = [getattr(arg, 'obj', arg) for arg in args]
        groupings = groupers[0].groupings
        return func(*args).groupby(groupings)

    # Define an execution rule for a simple elementwise Series
    # function
    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(pd.Series, nargs))
    )
    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(object, nargs))
    )
    def execute_udf_node(op, *args, **kwargs):
        # We have rewritten op.func to be a closure enclosing
        # the kwargs, and therefore, we do not need to pass
        # kwargs here. This is true for all udf execution in this
        # file.
        # See ibis.udf.vectorized.UserDefinedFunction
        return op.func(*args)

    return scope


@pre_execute.register(ops.AnalyticVectorizedUDF)
@pre_execute.register(ops.AnalyticVectorizedUDF, ibis.client.Client)
@pre_execute.register(ops.ReductionVectorizedUDF)
@pre_execute.register(ops.ReductionVectorizedUDF, ibis.client.Client)
def pre_execute_analytic_and_reduction_udf(op, *clients, scope=None, **kwargs):
    input_type = op.input_type
    nargs = len(input_type)

    # An execution rule to handle analytic and reduction UDFs over
    # 1) an ungrouped window,
    # 2) an ungrouped Aggregate node, or
    # 3) an ungrouped custom aggregation context
    @execute_node.register(type(op), *(itertools.repeat(pd.Series, nargs)))
    def execute_udaf_node_no_groupby(op, *args, aggcontext, **kwargs):
        return aggcontext.agg(args[0], op.func, *args[1:])

    # An execution rule to handle analytic and reduction UDFs over
    # 1) a grouped window,
    # 2) a grouped Aggregate node, or
    # 3) a grouped custom aggregation context
    @execute_node.register(type(op), *(itertools.repeat(SeriesGroupBy, nargs)))
    def execute_udaf_node_groupby(op, *args, aggcontext, **kwargs):
        func = op.func
        if isinstance(aggcontext, Transform):
            # We are either:
            # 1) Aggregating over an unbounded (and GROUPED) window, which
            #   uses a Transform aggregation context
            # 2) Aggregating using an Aggregate node (with GROUPING), which
            #   uses a Summarize aggregation context
            # We need to do some pre-processing to func and args so that
            # Transform or Summarize can pull data out of the SeriesGroupBys
            # in args.

            # Construct a generator that yields the next group of data
            # for every argument excluding the first (pandas performs
            # the iteration for the first argument) for each argument
            # that is a SeriesGroupBy.
            iters = create_gens_from_args_groupby(args[1:])

            # TODO: Unify calling convension here to be more like
            # window
            def aggregator(first, *rest):
                # map(next, *rest) gets the inputs for the next group
                # TODO: might be inefficient to do this on every call
                result = func(first, *map(next, rest))

                # Here we don't user execution.util.coerce_to_output
                # because this is the inner loop and we do not want
                # to wrap a scalar value with a series.
                if isinstance(op._output_type, dt.Struct):
                    return coerce_to_dataframe(result, op._output_type.names)
                else:
                    return result

            return aggcontext.agg(args[0], aggregator, *iters)
        elif isinstance(aggcontext, Summarize):
            iters = create_gens_from_args_groupby(args[1:])

            # TODO: Unify calling convension here to be more like
            # window
            def aggregator(first, *rest):
                # map(next, *rest) gets the inputs for the next group
                # TODO: might be inefficient to do this on every call
                return func(first, *map(next, rest))

            return aggcontext.agg(args[0], aggregator, *iters)
        else:
            # We are either:
            # 1) Aggregating over a bounded window, which uses a Window
            #  aggregation context
            # 2) Aggregating over a custom aggregation context
            # No pre-processing to be done for either case.
            return aggcontext.agg(args[0], func, *args[1:])

    return scope
