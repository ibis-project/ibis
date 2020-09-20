"""Dask backend execution of struct fields and literals."""

import collections
import operator

import dask.dataframe as dd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.expr.operations as ops
from ibis.dask.dispatch import execute_node


@execute_node.register(ops.StructField, collections.abc.Mapping)
def execute_node_struct_field_dict(op, data, **kwargs):
    return data[op.field]


@execute_node.register(ops.StructField, dd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    field = op.field
    # TODO This meta is not necessarily right
    return data.map(
        operator.itemgetter(field),
        meta=(data.name, data.dtype)
    ).rename(field)


# TODO - this is broken
@execute_node.register(ops.StructField, SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    field = op.field
    return (
        data.obj.map(operator.itemgetter(field))
        .rename(field)
        .groupby(data.grouper.groupings)
    )
