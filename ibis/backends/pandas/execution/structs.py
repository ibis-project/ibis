"""Pandas backend execution of struct fields and literals."""

import collections
import operator

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops

from ..dispatch import execute_node


@execute_node.register(ops.StructField, collections.abc.Mapping)
def execute_node_struct_field_dict(op, data, **kwargs):
    return data[op.field]


@execute_node.register(ops.StructField, pd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    field = op.field
    return data.map(operator.itemgetter(field)).rename(field)


@execute_node.register(ops.StructField, SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    field = op.field
    return (
        data.obj.map(operator.itemgetter(field))
        .rename(field)
        .groupby(data.grouper.groupings)
    )
