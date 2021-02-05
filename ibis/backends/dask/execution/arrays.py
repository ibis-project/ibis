import itertools

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.arrays import (
    execute_array_concat,
    execute_array_index,
    execute_array_length,
    execute_array_repeat,
    execute_array_slice,
)

from ..dispatch import execute_node
from .util import TypeRegistrationDict, register_types_to_dispatcher

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.ArrayLength: [((dd.Series,), execute_array_length)],
    ops.ArrayConcat: [
        ((dd.Series, (dd.Series, list)), execute_array_concat),
        ((list, dd.Series), execute_array_concat),
    ],
    ops.ArrayIndex: [((dd.Series, int), execute_array_index)],
    ops.ArrayRepeat: [
        ((dd.Series, dd.Series), execute_array_repeat),
        ((int, (dd.Series, list)), execute_array_repeat),
        (((dd.Series, list), int), execute_array_repeat),
    ],
    ops.ArraySlice: [
        ((dd.Series, int, (int, type(None))), execute_array_slice),
    ],
}

register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


collect_list = dd.Aggregation(
    name="collect_list",
    chunk=lambda s: s.apply(list),
    agg=lambda s0: s0.apply(
        lambda chunks: list(itertools.chain.from_iterable(chunks))
    ),
)


@execute_node.register(ops.ArrayColumn, list)
def execute_array_column(op, cols, **kwargs):
    df = dd.concat(cols, axis=1)
    return df.apply(lambda row: list(row), axis=1, meta=(None, 'object'))


# TODO - aggregations - #2553
@execute_node.register(ops.ArrayCollect, dd.Series)
def execute_array_collect(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, collect_list)


@execute_node.register(ops.ArrayCollect, ddgb.SeriesGroupBy)
def execute_array_collect_grouped_series(op, data, aggcontext=None, **kwargs):
    return data.agg(collect_list)
