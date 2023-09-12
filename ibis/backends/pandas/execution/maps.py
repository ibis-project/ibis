from __future__ import annotations

import collections
import functools

import numpy as np
import pandas as pd
import toolz

import ibis.expr.operations as ops
from ibis.backends.pandas.dispatch import execute_node


@execute_node.register(ops.Map, np.ndarray, np.ndarray)
def map_ndarray_ndarray(op, keys, values, **kwargs):
    return dict(zip(keys, values))


@execute_node.register(ops.Map, pd.Series, pd.Series)
def map_series_series(op, keys, values, **kwargs):
    return keys.combine(values, lambda a, b: dict(zip(a, b)))


@execute_node.register(ops.MapLength, pd.Series)
def map_length_series(op, data, **kwargs):
    # TODO: investigate whether calling a lambda is faster
    return data.dropna().map(len).reindex(data.index)


@execute_node.register(ops.MapLength, (collections.abc.Mapping, type(None)))
def map_length_dict(op, data, **kwargs):
    return None if data is None else len(data)


@execute_node.register(ops.MapGet, pd.Series, object, object)
def map_get_series_scalar_scalar(op, data, key, default, **kwargs):
    return data.map(functools.partial(safe_get, key=key, default=default))


@execute_node.register(ops.MapGet, pd.Series, object, pd.Series)
def map_get_series_scalar_series(op, data, key, default, **kwargs):
    defaultiter = iter(default.values)
    return data.map(
        lambda mapping, key=key, defaultiter=defaultiter: safe_get(
            mapping, key, next(defaultiter)
        )
    )


@execute_node.register(ops.MapGet, pd.Series, pd.Series, object)
def map_get_series_series_scalar(op, data, key, default, **kwargs):
    keyiter = iter(key.values)
    return data.map(
        lambda mapping, keyiter=keyiter, default=default: safe_get(
            mapping, next(keyiter), default
        )
    )


@execute_node.register(ops.MapGet, pd.Series, pd.Series, pd.Series)
def map_get_series_series_series(op, data, key, default):
    keyiter = iter(key.values)
    defaultiter = iter(default.values)

    def get(mapping, keyiter=keyiter, defaultiter=defaultiter):
        return safe_get(mapping, next(keyiter), next(defaultiter))

    return data.map(get)


@execute_node.register(ops.MapGet, collections.abc.Mapping, object, object)
def map_get_dict_scalar_scalar(op, data, key, default, **kwargs):
    return safe_get(data, key, default)


@execute_node.register(ops.MapGet, collections.abc.Mapping, object, pd.Series)
def map_get_dict_scalar_series(op, data, key, default, **kwargs):
    return default.map(lambda d, data=data, key=key: safe_get(data, key, d))


@execute_node.register(ops.MapGet, collections.abc.Mapping, pd.Series, object)
def map_get_dict_series_scalar(op, data, key, default, **kwargs):
    return key.map(lambda k, data=data, default=default: safe_get(data, k, default))


@execute_node.register(ops.MapGet, collections.abc.Mapping, pd.Series, pd.Series)
def map_get_dict_series_series(op, data, key, default, **kwargs):
    defaultiter = iter(default.values)
    return key.map(
        lambda k, data=data, defaultiter=defaultiter: safe_get(
            data, k, next(defaultiter)
        )
    )


@execute_node.register(ops.MapContains, collections.abc.Mapping, object)
def map_contains_dict_object(op, data, key, **kwargs):
    return safe_contains(data, key)


@execute_node.register(ops.MapContains, collections.abc.Mapping, pd.Series)
def map_contains_dict_series(op, data, key, **kwargs):
    return key.map(lambda k, data=data: safe_contains(data, k))


@execute_node.register(ops.MapContains, pd.Series, object)
def map_contains_series_object(op, data, key, **kwargs):
    return data.map(lambda d: safe_contains(d, key))


@execute_node.register(ops.MapContains, pd.Series, pd.Series)
def map_contains_series_series(op, data, key, **kwargs):
    return data.combine(key, lambda d, k: safe_contains(d, k))


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


@execute_node.register(ops.MapKeys, pd.Series)
def map_keys_series(op, data, **kwargs):
    return data.map(safe_keys)


@execute_node.register(ops.MapKeys, (collections.abc.Mapping, type(None)))
def map_keys_dict(op, data, **kwargs):
    if data is None:
        return None
    # list(...) to unpack iterable
    return np.array(list(data.keys()))


@execute_node.register(ops.MapValues, pd.Series)
def map_values_series(op, data, **kwargs):
    res = data.map(safe_values)
    return res


@execute_node.register(ops.MapValues, (collections.abc.Mapping, type(None)))
def map_values_dict(op, data, **kwargs):
    if data is None:
        return None
    # list(...) to unpack iterable
    return np.array(list(data.values()))


def safe_merge(*maps):
    return None if any(m is None for m in maps) else toolz.merge(*maps)


@execute_node.register(
    ops.MapMerge,
    (collections.abc.Mapping, type(None)),
    (collections.abc.Mapping, type(None)),
)
def map_merge_dict_dict(op, lhs, rhs, **kwargs):
    return safe_merge(lhs, rhs)


@execute_node.register(ops.MapMerge, (collections.abc.Mapping, type(None)), pd.Series)
def map_merge_dict_series(op, lhs, rhs, **kwargs):
    if lhs is None:
        return pd.Series([None] * len(rhs))
    return rhs.map(lambda m, lhs=lhs: safe_merge(lhs, m))


@execute_node.register(ops.MapMerge, pd.Series, (collections.abc.Mapping, type(None)))
def map_merge_series_dict(op, lhs, rhs, **kwargs):
    if rhs is None:
        return pd.Series([None] * len(lhs))
    return lhs.map(lambda m, rhs=rhs: safe_merge(m, rhs))


@execute_node.register(ops.MapMerge, pd.Series, pd.Series)
def map_merge_series_series(op, lhs, rhs, **kwargs):
    rhsiter = iter(rhs.values)
    return lhs.map(lambda m, rhsiter=rhsiter: safe_merge(m, next(rhsiter)))
