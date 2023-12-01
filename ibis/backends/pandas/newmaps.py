from __future__ import annotations

import collections
import functools

import numpy as np
import pandas as pd
import toolz

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import columnwise, elementwise, rowwise, serieswise


def safe_method(mapping, method, *args, **kwargs):
    if mapping is None:
        return None
    try:
        method = getattr(mapping, method)
    except AttributeError:
        return None
    else:
        return method(*args, **kwargs)


def safe_get(mapping, key, default=None):
    return safe_method(mapping, "get", key, default)


def safe_contains(mapping, key):
    return safe_method(mapping, "__contains__", key)


def safe_keys(mapping):
    result = safe_method(mapping, "keys")
    if result is None:
        return None
    # list(...) to unpack iterable
    return np.array(list(result))


def safe_values(mapping):
    result = safe_method(mapping, "values")
    if result is None:
        return None
    # list(...) to unpack iterable
    return np.array(list(result), dtype="object")


@execute.register(ops.MapLength)
def execute_map_length(op, **kwargs):
    return elementwise(len, **kwargs)


@execute.register(ops.MapKeys)
def execute_map_keys(op, **kwargs):
    return elementwise(safe_keys, **kwargs)


@execute.register(ops.MapValues)
def execute_map_values(op, **kwargs):
    return elementwise(safe_values, **kwargs)


@execute.register(ops.MapGet)
def execute_map_get(op, **kwargs):
    return rowwise(lambda row: safe_get(row["arg"], row["key"], row["default"]), kwargs)


@execute.register(ops.MapContains)
def execute_map_contains(op, **kwargs):
    return rowwise(lambda row: safe_contains(row["arg"], row["key"]), kwargs)


@execute.register(ops.Map)
def execute_map(op, **kwargs):
    return rowwise(lambda row: dict(zip(row["keys"], row["values"])), kwargs)


@execute.register(ops.MapMerge)
def execute_map_merge(op, **kwargs):
    return rowwise(lambda row: {**row["left"], **row["right"]}, kwargs)
