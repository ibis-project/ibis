import collections
import functools

import pandas as pd
import toolz

import ibis.expr.operations as ops

from ..dispatch import execute_node


@execute_node.register(ops.MapLength, pd.Series)
def execute_map_length_series(op, data, **kwargs):
    # TODO: investigate whether calling a lambda is faster
    return data.dropna().map(len).reindex(data.index)


@execute_node.register(ops.MapLength, (collections.abc.Mapping, type(None)))
def execute_map_length_dict(op, data, **kwargs):
    return None if data is None else len(data)


@execute_node.register(ops.MapValueForKey, pd.Series, pd.Series)
def execute_map_value_for_key_series_series(op, data, key, **kwargs):
    assert data.size == key.size, 'data.size != key.size'
    return data.map(
        lambda x, keyiter=iter(key.values): x.get(next(keyiter), None)
    )


@execute_node.register(ops.MapValueForKey, pd.Series, type(None))
def execute_map_value_for_key_series_none(op, data, key, **kwargs):
    return pd.Series([None] * len(data))


@execute_node.register(ops.MapValueForKey, pd.Series, object)
def execute_map_value_for_key_series_scalar(op, data, key, **kwargs):
    return data.map(functools.partial(safe_get, key=key))


@execute_node.register(ops.MapValueForKey, collections.abc.Mapping, pd.Series)
def execute_map_value_for_key_dict_series(op, data, key, **kwargs):
    return key.map(functools.partial(safe_get, data))


@execute_node.register(ops.MapValueForKey, collections.abc.Mapping, object)
def execute_map_value_for_key_dict_scalar(op, data, key, **kwargs):
    return safe_get(data, key)


@execute_node.register(ops.MapValueOrDefaultForKey, pd.Series, object, object)
def map_value_default_series_scalar_scalar(op, data, key, default, **kwargs):
    return data.map(functools.partial(safe_get, key=key, default=default))


@execute_node.register(
    ops.MapValueOrDefaultForKey, pd.Series, object, pd.Series
)
def map_value_default_series_scalar_series(op, data, key, default, **kwargs):
    return data.map(
        lambda mapping, key=key, defaultiter=iter(default.values): safe_get(
            mapping, key, next(defaultiter)
        )
    )


@execute_node.register(
    ops.MapValueOrDefaultForKey, pd.Series, pd.Series, object
)
def map_value_default_series_series_scalar(op, data, key, default, **kwargs):
    return data.map(
        lambda mapping, keyiter=iter(key.values), default=default: safe_get(
            mapping, next(keyiter), default
        )
    )


@execute_node.register(
    ops.MapValueOrDefaultForKey, pd.Series, pd.Series, pd.Series
)
def execute_map_value_default_series_series_series(op, data, key, default):
    def get(
        mapping, keyiter=iter(key.values), defaultiter=iter(default.values)
    ):
        return safe_get(mapping, next(keyiter), next(defaultiter))

    return data.map(get)


@execute_node.register(
    ops.MapValueOrDefaultForKey, collections.abc.Mapping, object, object
)
def execute_map_value_default_dict_scalar_scalar(
    op, data, key, default, **kwargs
):
    return safe_get(data, key, default)


@execute_node.register(
    ops.MapValueOrDefaultForKey, collections.abc.Mapping, object, pd.Series
)
def execute_map_value_default_dict_scalar_series(
    op, data, key, default, **kwargs
):
    return default.map(lambda d, data=data, key=key: safe_get(data, key, d))


@execute_node.register(
    ops.MapValueOrDefaultForKey, collections.abc.Mapping, pd.Series, object
)
def execute_map_value_default_dict_series_scalar(
    op, data, key, default, **kwargs
):
    return key.map(
        lambda k, data=data, default=default: safe_get(data, k, default)
    )


@execute_node.register(
    ops.MapValueOrDefaultForKey, collections.abc.Mapping, pd.Series, pd.Series
)
def execute_map_value_default_dict_series_series(
    op, data, key, default, **kwargs
):
    return key.map(
        lambda k, data=data, defaultiter=iter(default.values): safe_get(
            data, k, next(defaultiter)
        )
    )


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
    return safe_method(mapping, 'get', key, default)


def safe_keys(mapping):
    result = safe_method(mapping, 'keys')
    if result is None:
        return None
    return list(result)


def safe_values(mapping):
    result = safe_method(mapping, 'values')
    if result is None:
        return None
    return list(result)


@execute_node.register(ops.MapKeys, pd.Series)
def execute_map_keys_series(op, data, **kwargs):
    return data.map(safe_keys)


@execute_node.register(ops.MapKeys, (collections.abc.Mapping, type(None)))
def execute_map_keys_dict(op, data, **kwargs):
    if data is None:
        return None
    return list(data.keys())


@execute_node.register(ops.MapValues, pd.Series)
def execute_map_values_series(op, data, **kwargs):
    return data.map(safe_values)


@execute_node.register(ops.MapValues, (collections.abc.Mapping, type(None)))
def execute_map_values_dict(op, data, **kwargs):
    if data is None:
        return None
    return list(data.values())


def safe_merge(*maps):
    return None if any(m is None for m in maps) else toolz.merge(*maps)


@execute_node.register(
    ops.MapConcat,
    (collections.abc.Mapping, type(None)),
    (collections.abc.Mapping, type(None)),
)
def execute_map_concat_dict_dict(op, lhs, rhs, **kwargs):
    return safe_merge(lhs, rhs)


@execute_node.register(
    ops.MapConcat, (collections.abc.Mapping, type(None)), pd.Series
)
def execute_map_concat_dict_series(op, lhs, rhs, **kwargs):
    if lhs is None:
        return pd.Series([None] * len(rhs))
    return rhs.map(lambda m, lhs=lhs: safe_merge(lhs, m))


@execute_node.register(
    ops.MapConcat, pd.Series, (collections.abc.Mapping, type(None))
)
def execute_map_concat_series_dict(op, lhs, rhs, **kwargs):
    if rhs is None:
        return pd.Series([None] * len(lhs))
    return lhs.map(lambda m, rhs=rhs: safe_merge(m, rhs))


@execute_node.register(ops.MapConcat, pd.Series, pd.Series)
def execute_map_concat_series_series(op, lhs, rhs, **kwargs):
    return lhs.map(
        lambda m, rhsiter=iter(rhs.values): safe_merge(m, next(rhsiter))
    )
