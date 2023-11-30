from __future__ import annotations

import itertools
import operator

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import columnwise, elementwise, rowwise
from ibis.common.exceptions import OperationNotDefinedError


@execute.register(ops.ArrayLength)
def execute_array_length(op, arg):
    if isinstance(arg, pd.Series):
        return arg.apply(len)
    else:
        return len(arg)


@execute.register(ops.ArrayIndex)
def execute_array_index(op, **kwargs):
    def applier(row):
        try:
            return row["arg"][row["index"]]
        except IndexError:
            return None

    return rowwise(applier, kwargs)


@execute.register(ops.ArrayRepeat)
def execute_array_repeat(op, **kwargs):
    return rowwise(lambda row: np.tile(row["arg"], max(0, row["times"])), kwargs)


@execute.register(ops.ArraySlice)
def execute_array_slice(op, **kwargs):
    return rowwise(lambda row: row["arg"][row["start"] : row["stop"]], kwargs)


@execute.register(ops.ArrayColumn)
def execute_array_column(op, cols):
    return rowwise(lambda row: np.array(row, dtype=object), cols)


@execute.register(ops.ArrayFlatten)
def execute_array_flatten(op, arg):
    return elementwise(
        lambda v: list(itertools.chain.from_iterable(v)), arg, na_action="ignore"
    )


@execute.register(ops.ArrayConcat)
def execute_array_concat(op, arg):
    return rowwise(lambda row: np.concatenate(row.values), arg)


# @execute.register(ops.Unnest)
# def execute_unnest(op, arg):
#     return arg.explode()


# @execute_node.register(ops.ArrayContains, np.ndarray, object)
# def execute_node_contains_value_array(op, haystack, needle, **kwargs):
#     return needle in haystack
