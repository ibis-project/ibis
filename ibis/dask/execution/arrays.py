import itertools
import operator

import dask.dataframe as dd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.expr.operations as ops
from ibis.dask.dispatch import execute_node


@execute_node.register(ops.ArrayLength, dd.Series)
def execute_array_length(op, data, **kwargs):
    return data.apply(len)


@execute_node.register(ops.ArrayLength, list)
def execute_array_length_scalar(op, data, **kwargs):
    return len(data)


@execute_node.register(ops.ArraySlice, dd.Series, int, (int, type(None)))
def execute_array_slice(op, data, start, stop, **kwargs):
    return data.apply(operator.itemgetter(slice(start, stop)))


@execute_node.register(ops.ArraySlice, list, int, (int, type(None)))
def execute_array_slice_scalar(op, data, start, stop, **kwargs):
    return data[start:stop]


@execute_node.register(ops.ArrayIndex, dd.Series, int)
def execute_array_index(op, data, index, **kwargs):
    return data.apply(
        lambda array, index=index: (
            array[index] if -len(array) <= index < len(array) else None
        )
    )


@execute_node.register(ops.ArrayIndex, list, int)
def execute_array_index_scalar(op, data, index, **kwargs):
    try:
        return data[index]
    except IndexError:
        return None


@execute_node.register(ops.ArrayConcat, dd.Series, (dd.Series, list))
@execute_node.register(ops.ArrayConcat, list, dd.Series)
@execute_node.register(ops.ArrayConcat, list, list)
def execute_array_concat(op, left, right, **kwargs):
    return left + right


@execute_node.register(ops.ArrayRepeat, dd.Series, dd.Series)
@execute_node.register(ops.ArrayRepeat, int, (dd.Series, list))
@execute_node.register(ops.ArrayRepeat, (dd.Series, list), int)
def execute_array_repeat(op, left, right, **kwargs):
    return left * right


collect_list = dd.Aggregation(
    name="collect_list",
    chunk=lambda s: s.apply(list),
    agg=lambda s0: s0.apply(
        lambda chunks: list(itertools.chain.from_iterable(chunks))
    ),
)


@execute_node.register(ops.ArrayCollect, (dd.Series, SeriesGroupBy))
def execute_array_collect(op, data, aggcontext=None, **kwargs):

    return aggcontext.agg(data, collect_list)
