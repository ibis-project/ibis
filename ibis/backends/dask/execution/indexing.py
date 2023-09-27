"""Execution rules for ops.IfElse operations."""

from __future__ import annotations

import dask.dataframe as dd

import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.pandas.core import boolean_types, scalar_types, simple_types
from ibis.backends.pandas.execution.generic import pd_where


@execute_node.register(ops.IfElse, (dd.Series, *boolean_types), dd.Series, dd.Series)
@execute_node.register(ops.IfElse, (dd.Series, *boolean_types), dd.Series, simple_types)
@execute_node.register(ops.IfElse, (dd.Series, *boolean_types), simple_types, dd.Series)
@execute_node.register(ops.IfElse, (dd.Series, *boolean_types), type(None), type(None))
def execute_node_where(op, cond, true, false, **kwargs):
    if any(isinstance(x, (dd.Series, dd.core.Scalar)) for x in (cond, true, false)):
        return dd.map_partitions(pd_where, cond, true, false)
    # All are immediate scalars, handle locally
    return true if cond else false


# For true/false as scalars, we only support identical type pairs + None to
# limit the size of the dispatch table and not have to worry about type
# promotion.
for typ in (str, *scalar_types):
    for cond_typ in (dd.Series, *boolean_types):
        execute_node.register(ops.IfElse, cond_typ, typ, typ)(execute_node_where)
        execute_node.register(ops.IfElse, cond_typ, type(None), typ)(execute_node_where)
        execute_node.register(ops.IfElse, cond_typ, typ, type(None))(execute_node_where)
