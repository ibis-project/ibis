import collections

import dask.dataframe as dd
import numpy as np
import pandas

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.maps import (
    execute_map_keys_series,
    execute_map_value_default_dict_scalar_series,
    execute_map_value_default_dict_series_scalar,
    execute_map_value_default_dict_series_series,
    execute_map_value_default_series_series_series,
    execute_map_value_for_key_dict_series,
    execute_map_value_for_key_series_scalar,
    execute_map_values_series,
    map_value_default_series_scalar_scalar,
    map_value_default_series_scalar_series,
    map_value_default_series_series_scalar,
    safe_merge,
)

from ..dispatch import execute_node
from .util import TypeRegistrationDict, register_types_to_dispatcher

# NOTE - to avoid dispatch ambiguities we must unregister pandas, only to
# re-register below. The ordering in which dispatches are registered is
# meaningful. See https://multiple-dispatch.readthedocs.io/en/latest/resolution.html#ambiguities # noqa E501
# for more detail.
PANDAS_REGISTERED_TYPES = [
    (
        ops.MapValueOrDefaultForKey,
        collections.abc.Mapping,
        object,
        pandas.Series,
    ),
    (
        ops.MapValueOrDefaultForKey,
        collections.abc.Mapping,
        pandas.Series,
        object,
    ),
]
for registered_type in PANDAS_REGISTERED_TYPES:
    del execute_node[registered_type]


DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.MapValueForKey: [
        (
            (
                dd.Series,
                object,
            ),
            execute_map_value_for_key_series_scalar,
        ),
        (
            (
                collections.abc.Mapping,
                dd.Series,
            ),
            execute_map_value_for_key_dict_series,
        ),
    ],
    ops.MapValueOrDefaultForKey: [
        ((dd.Series, object, object), map_value_default_series_scalar_scalar),
        (
            (dd.Series, object, dd.Series),
            map_value_default_series_scalar_series,
        ),
        (
            (dd.Series, dd.Series, object),
            map_value_default_series_series_scalar,
        ),
        (
            (dd.Series, dd.Series, dd.Series),
            execute_map_value_default_series_series_series,
        ),
        # This never occurs but we need to register it so multipledispatch
        # does not see below registrations as ambigious. See NOTE above.
        (
            (
                collections.abc.Mapping,
                (dd.Series, pandas.Series),
                (dd.Series, pandas.Series),
            ),
            execute_map_value_default_dict_series_series,
        ),
        (
            (
                collections.abc.Mapping,
                object,
                (dd.Series, pandas.Series),
            ),
            execute_map_value_default_dict_scalar_series,
        ),
        (
            (
                collections.abc.Mapping,
                (dd.Series, pandas.Series),
                object,
            ),
            execute_map_value_default_dict_series_scalar,
        ),
    ],
    ops.MapKeys: [((dd.Series,), execute_map_keys_series)],
    ops.MapValues: [((dd.Series,), execute_map_values_series)],
}
register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


@execute_node.register(ops.MapLength, dd.Series)
def execute_map_length_series(op, data, **kwargs):
    return data.map(len, na_action='ignore')


@execute_node.register(ops.MapValueForKey, dd.Series, dd.Series)
def execute_map_value_for_key_series_series(op, data, key, **kwargs):
    assert data.size == key.size, 'data.size != key.size'
    return data.map(
        lambda x, keyiter=iter(key.values): x.get(next(keyiter), None)
    )


def none_filled_dask_series(n):
    dd.from_array(np.full(n, None))


@execute_node.register(ops.MapValueForKey, dd.Series, type(None))
def execute_map_value_for_key_series_none(op, data, key, **kwargs):
    return none_filled_dask_series(len(data))


@execute_node.register(
    ops.MapConcat, (collections.abc.Mapping, type(None)), dd.Series
)
def execute_map_concat_dict_series(op, lhs, rhs, **kwargs):
    if lhs is None:
        return none_filled_dask_series(len(rhs))
    return rhs.map(
        lambda m, lhs=lhs: safe_merge(lhs, m),
        meta=(rhs.name, rhs.dtype),
    )


@execute_node.register(
    ops.MapConcat, dd.Series, (collections.abc.Mapping, type(None))
)
def execute_map_concat_series_dict(op, lhs, rhs, **kwargs):
    if rhs is None:
        return none_filled_dask_series(len(lhs))
    return lhs.map(
        lambda m, rhs=rhs: safe_merge(m, rhs),
        meta=(lhs.name, lhs.dtype),
    )


@execute_node.register(ops.MapConcat, dd.Series, dd.Series)
def execute_map_concat_series_series(op, lhs, rhs, **kwargs):
    return lhs.map(
        lambda m, rhsiter=iter(rhs.values): safe_merge(m, next(rhsiter)),
        meta=(lhs.name, lhs.dtype),
    )
