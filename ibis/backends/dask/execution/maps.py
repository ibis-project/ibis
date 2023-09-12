from __future__ import annotations

from collections.abc import Mapping

import dask.dataframe as dd
import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    TypeRegistrationDict,
    register_types_to_dispatcher,
)
from ibis.backends.pandas.execution.maps import (
    map_contains_dict_series,
    map_contains_series_object,
    map_contains_series_series,
    map_get_dict_scalar_series,
    map_get_dict_series_scalar,
    map_get_dict_series_series,
    map_get_series_scalar_scalar,
    map_get_series_scalar_series,
    map_get_series_series_scalar,
    map_get_series_series_series,
    map_keys_series,
    map_series_series,
    map_values_series,
    safe_merge,
)

# NOTE - to avoid dispatch ambiguities we must unregister pandas, only to
# re-register below. The ordering in which dispatches are registered is
# meaningful. See
# https://multiple-dispatch.readthedocs.io/en/latest/resolution.html#ambiguities
# for more detail.
PANDAS_REGISTERED_TYPES = [
    (ops.MapGet, Mapping, object, pd.Series),
    (ops.MapGet, Mapping, pd.Series, object),
]
for registered_type in PANDAS_REGISTERED_TYPES:
    del execute_node[registered_type]


DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.Map: [((dd.Series, dd.Series), map_series_series)],
    ops.MapGet: [
        ((dd.Series, object, object), map_get_series_scalar_scalar),
        ((dd.Series, object, dd.Series), map_get_series_scalar_series),
        ((dd.Series, dd.Series, object), map_get_series_series_scalar),
        ((dd.Series, dd.Series, dd.Series), map_get_series_series_series),
        # This never occurs but we need to register it so multipledispatch
        # does not see below registrations as ambiguous. See NOTE above.
        (
            (Mapping, (dd.Series, pd.Series), (dd.Series, pd.Series)),
            map_get_dict_series_series,
        ),
        ((Mapping, object, (dd.Series, pd.Series)), map_get_dict_scalar_series),
        ((Mapping, (dd.Series, pd.Series), object), map_get_dict_series_scalar),
    ],
    ops.MapContains: [
        ((Mapping, dd.Series), map_contains_dict_series),
        ((dd.Series, dd.Series), map_contains_series_series),
        ((dd.Series, object), map_contains_series_object),
    ],
    ops.MapKeys: [((dd.Series,), map_keys_series)],
    ops.MapValues: [((dd.Series,), map_values_series)],
}
register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


@execute_node.register(ops.MapLength, dd.Series)
def map_length_series(op, data, **kwargs):
    return data.map(len, na_action="ignore")


def none_filled_dask_series(n):
    dd.from_array(np.full(n, None))


@execute_node.register(ops.MapMerge, (Mapping, type(None)), dd.Series)
def execute_map_concat_dict_series(op, lhs, rhs, **kwargs):
    if lhs is None:
        return none_filled_dask_series(len(rhs))
    return rhs.map(
        lambda m, lhs=lhs: safe_merge(lhs, m),
        meta=(rhs.name, rhs.dtype),
    )


@execute_node.register(ops.MapMerge, dd.Series, (Mapping, type(None)))
def execute_map_concat_series_dict(op, lhs, rhs, **kwargs):
    if rhs is None:
        return none_filled_dask_series(len(lhs))
    return lhs.map(
        lambda m, rhs=rhs: safe_merge(m, rhs),
        meta=(lhs.name, lhs.dtype),
    )


@execute_node.register(ops.MapMerge, dd.Series, dd.Series)
def execute_map_concat_series_series(op, lhs, rhs, **kwargs):
    rhsiter = iter(rhs.values)
    return lhs.map(
        lambda m, rhsiter=rhsiter: safe_merge(m, next(rhsiter)),
        meta=(lhs.name, lhs.dtype),
    )
