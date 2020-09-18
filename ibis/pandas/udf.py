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

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.udf.vectorized
from ibis.pandas.aggcontext import Summarize, Transform, Window
from ibis.pandas.core import (
    date_types,
    time_types,
    timedelta_types,
    timestamp_types,
)
from ibis.pandas.dispatch import execute_node, pre_execute


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

    # An execution rule to handle analytic and reduction UDFs over an
    # ungrouped window or an ungrouped Aggregate node
    @execute_node.register(type(op), *(itertools.repeat(pd.Series, nargs)))
    def execute_udaf_node_no_groupby(op, *args, **kwargs):
        aggcontext = kwargs.pop('aggcontext', None)
        assert aggcontext is not None, 'aggcontext is None'

        func = op.func

        if isinstance(aggcontext, Summarize) or isinstance(
            aggcontext, Transform
        ):
            # We are either:
            # 1. Aggregating over an unbounded (and ungrouped) window
            # 2. Aggregating using an Aggregate node (with no grouping)
            # No information from aggcontext is needed in either scenario.
            # Call func directly.
            return func(*args)
        elif isinstance(aggcontext, Window):
            # We must be aggregating over a bounded window, so we need
            # aggcontext.agg to handle the rolling/expanding window logic.
            return aggcontext.agg(args[0], func, *args, **kwargs)
        else:
            # We must be aggregating over a custom AggregationContext. We'll
            # call its .agg method and let it handle any further custom logic.
            return aggcontext.agg(args[0], func, *args, **kwargs)

    # An execution rule to handle analytic and reduction UDFs over
    # a grouped window or a grouped Aggregate node
    @execute_node.register(type(op), *(itertools.repeat(SeriesGroupBy, nargs)))
    def execute_udaf_node_groupby(op, *args, **kwargs):
        aggcontext = kwargs.pop('aggcontext', None)
        assert aggcontext is not None, 'aggcontext is None'

        func = op.func

        if isinstance(aggcontext, Summarize) or isinstance(
            aggcontext, Transform
        ):
            # We are either:
            # 1. Aggregating over an unbounded (and GROUPED) window
            # 2. Aggregating using an Aggregate node (with GROUPING)
            # In either case, we need to unpack the data from the
            # SeriesGroupBys then use aggcontext.agg to handle the
            # grouped aggregation logic.

            # Construct a generator that yields the next group of data
            # for every argument excluding the first (pandas performs
            # the iteration for the first argument) for each argument
            # that is a SeriesGroupBy.
            iters = create_gens_from_args_groupby(args[1:])

            # TODO: Unify calling convension here to be more like
            # window
            def aggregator(first, *rest, **kwargs):
                # map(next, *rest) gets the inputs for the next group
                # TODO: might be inefficient to do this on every call
                return func(first, *map(next, rest))

            return aggcontext.agg(args[0], aggregator, *iters, **kwargs)
        elif isinstance(aggcontext, Window):
            # We must be aggregating over a bounded window, so we need
            # aggcontext.agg to handle the rolling/expanding window logic.
            return aggcontext.agg(args[0], func, *args, **kwargs)
        else:
            # We must be aggregating over a custom AggregationContext. We'll
            # call its .agg method and let it handle any further custom logic.
            return aggcontext.agg(args[0], func, *args, **kwargs)

    return scope
