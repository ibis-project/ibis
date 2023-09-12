"""Dask backend execution of struct fields and literals."""

from __future__ import annotations

import operator

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb

import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import make_selected_obj


@execute_node.register(ops.StructField, dd.DataFrame)
def execute_node_struct_field_dict(op, data, **kwargs):
    return data[op.field]


@execute_node.register(ops.StructField, dd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    # TODO This meta is not necessarily right
    getter = operator.itemgetter(op.field)
    return data.map(getter, meta=(data.name, data.dtype)).rename(op.field)


@execute_node.register(ops.StructField, ddgb.SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    selected_obj = make_selected_obj(data)
    getter = operator.itemgetter(op.field)
    return (
        selected_obj.map(getter, meta=selected_obj._meta)
        .rename(op.field)
        .groupby(data.index)
    )
