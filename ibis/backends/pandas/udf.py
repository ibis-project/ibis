"""APIs for creating user-defined functions."""

from __future__ import annotations

import itertools

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops
import ibis.legacy.udf.vectorized
from ibis.backends.base import BaseBackend
from ibis.backends.pandas.aggcontext import Transform
from ibis.backends.pandas.dispatch import execute_node, pre_execute
from ibis.backends.pandas.execution.util import get_grouping


def create_gens_from_args_groupby(*args: tuple[SeriesGroupBy, ...]):
    """Create generators for each of `args` for groupby UDAF.

    Returns a generator that outputs each group.

    Parameters
    ----------
    *args
        A tuple of group by objects

    Returns
    -------
    Tuple[Generator]
        Generators of group by data
    """
    return ((data for _, data in arg) for arg in args)


class udf:
    @staticmethod
    def elementwise(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.elementwise."""

        return ibis.legacy.udf.vectorized.elementwise(input_type, output_type)

    @staticmethod
    def reduction(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.reduction."""
        return ibis.legacy.udf.vectorized.reduction(input_type, output_type)

    @staticmethod
    def analytic(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.analytic."""
        return ibis.legacy.udf.vectorized.analytic(input_type, output_type)


@pre_execute.register(ops.ElementWiseVectorizedUDF)
@pre_execute.register(ops.ElementWiseVectorizedUDF, BaseBackend)
def pre_execute_elementwise_udf(op, *clients, scope=None, **kwargs):
    """Register execution rules for elementwise UDFs."""
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
            for grouper in (getattr(arg, "grouper", None) for arg in args)
            if grouper is not None
        ]

        # all grouping keys must be identical
        assert all(groupers[0] == grouper for grouper in groupers[1:])

        # we're performing a scalar operation on grouped column, so
        # perform the operation directly on the underlying Series
        # and regroup after it's finished
        args = [getattr(arg, "obj", arg) for arg in args]
        groupings = get_grouping(groupers[0].groupings)
        return func(*args).groupby(groupings, group_keys=False)

    # Define an execution rule for a simple elementwise Series
    # function
    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(pd.Series, nargs))
    )
    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(object, nargs))
    )
    def execute_udf_node(op, *args, cache=None, timecontext=None, **kwargs):
        # We have rewritten op.func to be a closure enclosing
        # the kwargs, and therefore, we do not need to pass
        # kwargs here. This is true for all udf execution in this
        # file.
        # See ibis.legacy.udf.vectorized.UserDefinedFunction

        # prevent executing UDFs multiple times on different execution branches
        try:
            result = cache[(op, timecontext)]
        except KeyError:
            result = cache[(op, timecontext)] = op.func(*args)

        return result

    return scope


@pre_execute.register(ops.AnalyticVectorizedUDF)
@pre_execute.register(ops.AnalyticVectorizedUDF, BaseBackend)
@pre_execute.register(ops.ReductionVectorizedUDF)
@pre_execute.register(ops.ReductionVectorizedUDF, BaseBackend)
def pre_execute_analytic_and_reduction_udf(op, *clients, scope=None, **kwargs):
    input_type = op.input_type
    nargs = len(input_type)

    # An execution rule to handle analytic and reduction UDFs over
    # 1) an ungrouped window,
    # 2) an ungrouped Aggregate node, or
    # 3) an ungrouped custom aggregation context
    @execute_node.register(type(op), *(itertools.repeat(pd.Series, nargs)))
    def execute_udaf_node_no_groupby(op, *args, aggcontext, **kwargs):
        func = op.func
        return aggcontext.agg(args[0], func, *args[1:])

    # An execution rule to handle analytic and reduction UDFs over
    # 1) a grouped window,
    # 2) a grouped Aggregate node, or
    # 3) a grouped custom aggregation context
    @execute_node.register(type(op), *(itertools.repeat(SeriesGroupBy, nargs)))
    def execute_udaf_node_groupby(op, *args, aggcontext, **kwargs):
        func = op.func
        if isinstance(aggcontext, Transform):
            # We are aggregating over an unbounded (and GROUPED) window,
            # which uses a Transform aggregation context.
            # We need to do some pre-processing to func and args so that
            # Transform can pull data out of the SeriesGroupBys in args.

            # Construct a generator that yields the next group of data
            # for every argument excluding the first (pandas performs
            # the iteration for the first argument) for each argument
            # that is a SeriesGroupBy.
            iters = create_gens_from_args_groupby(*args[1:])

            # TODO: Unify calling convention here to be more like
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
            # 3) Aggregating using an Aggregate node (with GROUPING), which
            #   uses a Summarize aggregation context
            # No pre-processing to be done for any case.
            return aggcontext.agg(args[0], func, *args[1:])

    return scope
