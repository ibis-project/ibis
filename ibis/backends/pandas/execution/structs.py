"""Pandas backend execution of struct fields and literals."""

import collections
import operator

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops

from ..dispatch import execute_node


@execute_node.register(
    ops.StructField, (collections.abc.Mapping, pd.DataFrame)
)
def execute_node_struct_field_dict(op, data, **kwargs):
    return data[op.field]


@execute_node.register(ops.StructField, pd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    getter = operator.itemgetter(op.field)
    return data.map(getter).rename(op.field)


@execute_node.register(ops.StructField, SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    getter = operator.itemgetter(op.field)
    return (
        data.obj.map(getter).rename(op.field).groupby(data.grouper.groupings)
    )
