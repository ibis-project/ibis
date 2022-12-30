"""Pandas backend execution of struct fields and literals."""

from __future__ import annotations

import collections
import functools

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops
from ibis.backends.pandas.dispatch import execute_node
from ibis.backends.pandas.execution.util import get_grouping


@execute_node.register(ops.StructField, (collections.abc.Mapping, pd.DataFrame))
def execute_node_struct_field_dict(op, data, **kwargs):
    return data[op.field]


@execute_node.register(ops.StructField, (type(None), type(pd.NA), float))
def execute_node_struct_field_none(op, data, **_):
    assert (isinstance(data, float) and pd.isna(data)) or not isinstance(data, float)
    return pd.NA


def _safe_getter(value, field: str):
    if pd.isna(value):
        return pd.NA
    else:
        return value[field]


@execute_node.register(ops.StructField, pd.Series)
def execute_node_struct_field_series(op, data, **kwargs):
    getter = functools.partial(_safe_getter, field=op.field)
    return data.map(getter).rename(op.field)


@execute_node.register(ops.StructField, SeriesGroupBy)
def execute_node_struct_field_series_group_by(op, data, **kwargs):
    getter = functools.partial(_safe_getter, field=op.field)
    groupings = get_grouping(data.grouper.groupings)
    return data.obj.map(getter).rename(op.field).groupby(groupings, group_keys=False)
