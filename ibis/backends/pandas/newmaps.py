from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import elementwise, rowwise


def safe_method(mapping, method, *args, **kwargs):
    if mapping is None or mapping is pd.NA:
        return mapping
    try:
        method = getattr(mapping, method)
    except AttributeError:
        return None
    else:
        return method(*args, **kwargs)


def safe_len(mapping):
    return safe_method(mapping, "__len__")


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


def safe_merge(left, right):
    if left is None or left is pd.NA:
        return None
    elif right is None or right is pd.NA:
        return None
    else:
        return {**left, **right}


@execute.register(ops.MapLength)
def execute_map_length(op, **kwargs):
    return elementwise(safe_len, **kwargs)


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
    return rowwise(lambda row: safe_merge(row["left"], row["right"]), kwargs)


@execute.register(ops.StructField)
def execute_struct_field(op, arg, field):
    return elementwise(lambda x: safe_get(x, field), arg=arg)


def safe_json_getitem(value, key):
    try:
        # try to deserialize the value -> return None if it's None
        if (js := json.loads(value)) is None:
            return None
    except (json.JSONDecodeError, TypeError):
        # if there's an error related to decoding or a type error return None
        return None

    try:
        # try to extract the value as an array element or mapping key
        return json.dumps(js[key])
    except (KeyError, IndexError, TypeError):
        # KeyError: missing mapping key
        # IndexError: missing sequence key
        # TypeError: `js` doesn't implement __getitem__, either at all or for
        # the type of `key`
        return None


@execute.register(ops.JSONGetItem)
def execute_json_getitem(op, **kwargs):
    return rowwise(lambda row: safe_json_getitem(row["arg"], row["index"]), kwargs)
