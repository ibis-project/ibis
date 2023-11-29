from __future__ import annotations

import functools
import itertools

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import Cast

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    geospatial_functions,
    get_col,
    get_sqla_table,
    reduction,
    unary,
    varargs,
)
from ibis.backends.postgres.registry import operation_registry as _operation_registry

operation_registry = {
    op: _operation_registry[op]
    for op in _operation_registry.keys() - geospatial_functions.keys()
}


# def _unnest(t, op):
#     arg = t.translate(op.arg)
#     # HACK: https://community.snowflake.com/s/question/0D50Z000086MVhnSAG/has-anyone-found-a-way-to-unnest-an-array-without-loosing-the-null-values
#     sep = util.guid()
#     col = sa.func.nullif(
#         sa.func.split_to_table(sa.func.array_to_string(arg, sep), sep)
#         .table_valued("value")  # seq, index, value is supported but we only need value
#         .lateral()
#         .c["value"],
#         "",
#     )
#     return sa.cast(
#         sa.func.coalesce(sa.func.try_parse_json(col), sa.func.to_variant(col)),
#         type_=t.get_sqla_type(op.dtype),
#     )


operation_registry.update(
    {
        # numbers
        # columns
        ops.Unnest: _unnest,
    }
)

_invalid_operations = {
    # ibis.expr.operations.array
    ops.ArrayMap,
    ops.ArrayFilter,
    # ibis.expr.operations.reductions
    ops.MultiQuantile,
    # ibis.expr.operations.strings
    ops.FindInSet,
    # ibis.expr.operations.temporal
    ops.IntervalFromInteger,
    ops.TimestampDiff,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
