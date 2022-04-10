"""Execution rules for Aggregatons - mostly TODO

- ops.Aggregation
- ops.Any
- ops.NotAny
- ops.All
- ops.NotAll

"""

import functools
import operator
from typing import Optional

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb

import ibis.expr.operations as ops
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import coerce_to_output, safe_concat
from ibis.backends.pandas.execution.generic import agg_ctx
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext


# TODO - aggregations - #2553
# Not all code paths work cleanly here
@execute_node.register(
    ops.Aggregation,
    dd.DataFrame,
    tuple,
    tuple,
    tuple,
    tuple,
    tuple,
)
def execute_aggregation_dataframe(
    op,
    data,
    metrics,
    by,
    having,
    predicates,
    sort_keys,
    scope=None,
    timecontext: Optional[TimeContext] = None,
    **kwargs
):
    assert metrics, 'no metrics found during aggregation execution'

    if sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    if predicates:
        predicate = functools.reduce(
            operator.and_,
            (
                execute(p, scope=scope, timecontext=timecontext, **kwargs)
                for p in predicates
            ),
        )
        data = data.loc[predicate]

    columns = {}

    if by:
        grouping_key_pairs = list(
            zip(by, map(operator.methodcaller('op'), by))
        )
        grouping_keys = [
            by_op.name
            if isinstance(by_op, ops.TableColumn)
            else execute(
                by, scope=scope, timecontext=timecontext, **kwargs
            ).rename(by.get_name())
            for by, by_op in grouping_key_pairs
        ]
        columns.update(
            (by_op.name, by.get_name())
            for by, by_op in grouping_key_pairs
            if hasattr(by_op, 'name')
        )
        source = data.groupby(grouping_keys)
    else:
        source = data

    scope = scope.merge_scope(Scope({op.table.op(): source}, timecontext))

    pieces = []
    for metric in metrics:
        piece = execute(metric, scope=scope, timecontext=timecontext, **kwargs)
        piece = coerce_to_output(piece, metric)
        pieces.append(piece)

    # We must perform this check here otherwise dask will throw a ValueError
    # on `concat_and_check`. See docstring on `util.concat_via_join` for
    # more detail
    result = safe_concat(pieces)

    # If grouping, need a reset to get the grouping key back as a column
    if by:
        result = result.reset_index()

    result.columns = [columns.get(c, c) for c in result.columns]

    if having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not by:
            raise ValueError(
                'Filtering out aggregation values is not allowed without at '
                'least one grouping key'
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (
                execute(having, scope=scope, timecontext=timecontext, **kwargs)
                for having in having
            ),
        )
        assert len(predicate) == len(
            result
        ), 'length of predicate does not match length of DataFrame'
        result = result.loc[predicate.values]
    return result


@execute_node.register((ops.Any, ops.All), (dd.Series, ddgb.SeriesGroupBy))
def execute_any_all_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = aggcontext.agg(data, type(op).__name__.lower())
    else:
        # Note this branch is not currently hit in the dask backend but is
        # here for future scafolding.
        result = aggcontext.agg(
            data, lambda data: getattr(data, type(op).__name__.lower())()
        )
    return result


@execute_node.register(ops.NotAny, (dd.Series, ddgb.SeriesGroupBy))
def execute_notany_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = ~(aggcontext.agg(data, 'any'))
    else:
        # Note this branch is not currently hit in the dask backend but is
        # here for future scafolding.
        result = aggcontext.agg(data, lambda data: ~(data.any()))

    return result


@execute_node.register(ops.NotAll, (dd.Series, ddgb.SeriesGroupBy))
def execute_notall_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = ~(aggcontext.agg(data, 'all'))
    else:
        # Note this branch is not currently hit in the dask backend but is
        # here for future scafolding.
        result = aggcontext.agg(data, lambda data: ~(data.all()))

    return result
