"""Dask backend execution of struct fields and literals."""

import operator

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb

import ibis.expr.operations as ops

from ..dispatch import execute_node
from .util import make_selected_obj


@execute_node.register(ops.StructField, dd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    field = op.field
    # TODO This meta is not necessarily right
    return data.map(
        operator.itemgetter(field), meta=(data.name, data.dtype)
    ).rename(field)


@execute_node.register(ops.StructField, ddgb.SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    field = op.field
    selected_obj = make_selected_obj(data)
    return (
        selected_obj.map(operator.itemgetter(field), meta=selected_obj._meta)
        .rename(field)
        .groupby(data.index)
    )
