"""Execution rules for ops.Where operations"""

import dask.dataframe as dd
import numpy as np

import ibis.expr.operations as ops
from ibis.backends.pandas.core import boolean_types, scalar_types
from ibis.backends.pandas.execution.generic import (
    execute_node_where_scalar_scalar_scalar,
    execute_node_where_series_series_series,
)

from ..dispatch import execute_node
from .util import TypeRegistrationDict, register_types_to_dispatcher

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.Where: [
        (
            (dd.Series, dd.Series, dd.Series),
            execute_node_where_series_series_series,
        ),
        (
            (dd.Series, dd.Series, scalar_types),
            execute_node_where_series_series_series,
        ),
        (
            (
                boolean_types,
                dd.Series,
                dd.Series,
            ),
            execute_node_where_scalar_scalar_scalar,
        ),
    ]
}
register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


def execute_node_where_series_scalar_scalar(op, cond, true, false, **kwargs):
    return dd.from_array(np.repeat(true, len(cond))).where(cond, other=false)


for scalar_type in scalar_types:
    execute_node.register(ops.Where, dd.Series, scalar_type, scalar_type)(
        execute_node_where_series_scalar_scalar
    )


@execute_node.register(ops.Where, boolean_types, dd.Series, scalar_types)
def execute_node_where_scalar_series_scalar(op, cond, true, false, **kwargs):
    if cond:
        return true
    else:
        # TODO double check this is the right way to do this
        out = dd.from_array(np.repeat(false, len(true)))
        out.index = true.index
        return out


@execute_node.register(ops.Where, boolean_types, scalar_types, dd.Series)
def execute_node_where_scalar_scalar_series(op, cond, true, false, **kwargs):
    return dd.from_array(np.repeat(true, len(false))) if cond else false
