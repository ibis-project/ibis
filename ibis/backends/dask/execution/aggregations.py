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

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.generic import execute, execute_node
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext


# TODO - aggregations - #2553
# Not all code paths work cleanly here
@execute_node.register(ops.Aggregation, dd.DataFrame)
def execute_aggregation_dataframe(
    op, data, scope=None, timecontext: Optional[TimeContext] = None, **kwargs
):
    assert op.metrics, 'no metrics found during aggregation execution'

    if op.sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    predicates = op.predicates
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

    if op.by:
        grouping_key_pairs = list(
            zip(op.by, map(operator.methodcaller('op'), op.by))
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
    for metric in op.metrics:
        piece = execute(metric, scope=scope, timecontext=timecontext, **kwargs)
        piece.name = metric.get_name()
        pieces.append(piece)

    result = dd.concat(pieces, axis=1)

    # If grouping, need a reset to get the grouping key back as a column
    if op.by:
        result = result.reset_index()

    result.columns = [columns.get(c, c) for c in result.columns]

    if op.having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not op.by:
            raise ValueError(
                'Filtering out aggregation values is not allowed without at '
                'least one grouping key'
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (
                execute(having, scope=scope, timecontext=timecontext, **kwargs)
                for having in op.having
            ),
        )
        assert len(predicate) == len(
            result
        ), 'length of predicate does not match length of DataFrame'
        result = result.loc[predicate.values]
    return result


# TODO - aggregations - #2553
# @execute_node.register((ops.Any, ops.All), (dd.Series, SeriesGroupBy))
# def execute_any_all_series(op, data, aggcontext=None, **kwargs):
#     if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
#         result = aggcontext.agg(data, type(op).__name__.lower())
#     else:
#         result = aggcontext.agg(
#             data, lambda data: getattr(data, type(op).__name__.lower())()
#         )
#     return result

# TODO - aggregations - #2553
# @execute_node.register(ops.NotAny, (dd.Series, SeriesGroupBy))
# def execute_notany_series(op, data, aggcontext=None, **kwargs):
#     if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
#         result = ~(aggcontext.agg(data, 'any'))
#     else:
#         result = aggcontext.agg(data, lambda data: ~(data.any()))
#     try:
#         return result.astype(bool)
#     except TypeError:
#         return result

# TODO - aggregations - #2553
# @execute_node.register(ops.NotAll, (dd.Series, SeriesGroupBy))
# def execute_notall_series(op, data, aggcontext=None, **kwargs):
#     if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
#         result = ~(aggcontext.agg(data, 'all'))
#     else:
#         result = aggcontext.agg(data, lambda data: ~(data.all()))
#     try:
#         return result.astype(bool)
#     except TypeError:
#         return result
