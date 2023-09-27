"""Methods to translate BigQuery expressions before compilation."""

from __future__ import annotations

import toolz

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql import compiler as sql_compiler


def bq_sum(op):
    if isinstance((arg := op.arg).dtype, dt.Boolean):
        return ops.Sum(ops.Cast(arg, dt.int64), where=op.where)
    else:
        return op


def bq_mean(op):
    if isinstance((arg := op.arg).dtype, dt.Boolean):
        return ops.Mean(ops.Cast(arg, dt.int64), where=op.where)
    else:
        return op


REWRITES = {
    **sql_compiler.ExprTranslator._rewrites,
    ops.Sum: bq_sum,
    ops.Mean: bq_mean,
    ops.Any: toolz.identity,
    ops.All: toolz.identity,
}
