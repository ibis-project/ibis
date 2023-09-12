from __future__ import annotations

import itertools
from functools import partial

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np

import ibis.expr.operations as ops
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    TypeRegistrationDict,
    register_types_to_dispatcher,
)
from ibis.backends.pandas.execution.arrays import (
    execute_array_index,
    execute_array_length,
)

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.ArrayLength: [((dd.Series,), execute_array_length)],
    ops.ArrayIndex: [((dd.Series, int), execute_array_index)],
}

register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


collect_list = dd.Aggregation(
    name="collect_list",
    chunk=lambda s: s.apply(list),
    agg=lambda s0: s0.apply(lambda chunks: list(itertools.chain.from_iterable(chunks))),
)


@execute_node.register(ops.ArrayColumn, tuple)
def execute_array_column(op, cols, **kwargs):
    cols = [execute(arg, **kwargs) for arg in cols]
    df = dd.concat(cols, axis=1)
    return df.apply(
        lambda row: np.array(row, dtype=object), axis=1, meta=(None, "object")
    )


# TODO - aggregations - #2553
@execute_node.register(ops.ArrayCollect, dd.Series, type(None))
def execute_array_collect(op, data, where, aggcontext=None, **kwargs):
    return aggcontext.agg(data, collect_list)


@execute_node.register(ops.ArrayCollect, ddgb.SeriesGroupBy, type(None))
def execute_array_collect_grouped_series(op, data, where, **kwargs):
    return data.agg(collect_list)


@execute_node.register(ops.ArrayConcat, tuple)
def execute_array_concat(op, args, **kwargs):
    return execute_node(op, *map(partial(execute, **kwargs), args), **kwargs)
