"""Dask backend execution of struct fields and literals."""

import operator

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.structs import execute_node


@execute_node.register(ops.StructField, dd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    field = op.field
    # TODO This meta is not necessarily right
    return data.map(
        operator.itemgetter(field), meta=(data.name, data.dtype)
    ).rename(field)


# TODO - grouping - #2553
@execute_node.register(ops.StructField, ddgb.SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    field = op.field
    return (
        data.obj.map(operator.itemgetter(field))
        .rename(field)
        .groupby(data.grouper.groupings)
    )
