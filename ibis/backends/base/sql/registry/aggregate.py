from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


def _maybe_cast_bool(translator, op, arg):
    if (
        translator._bool_aggs_need_cast_to_int32
        and isinstance(op, (ops.Sum, ops.Mean, ops.Min, ops.Max))
        and (dtype := arg.dtype).is_boolean()
    ):
        return ops.Cast(arg, dt.Int32(nullable=dtype.nullable))
    return arg


def _reduction_format(translator, op, func_name, where, *args):
    args = (
        _maybe_cast_bool(translator, op, arg)
        for arg in args
        if isinstance(arg, ops.Node)
    )
    if where is not None:
        args = (ops.IfElse(where, arg, ibis.NA) for arg in args)

    return "{}({})".format(
        func_name,
        ", ".join(map(translator.translate, args)),
    )


def reduction(func_name):
    def formatter(translator, op):
        *args, where = op.args
        return _reduction_format(translator, op, func_name, where, *args)

    return formatter


def variance_like(func_name):
    func_names = {
        "sample": f"{func_name}_samp",
        "pop": f"{func_name}_pop",
    }

    def formatter(translator, op):
        return _reduction_format(translator, op, func_names[op.how], op.where, op.arg)

    return formatter


def count_distinct(translator, op):
    if op.where is not None:
        arg_formatted = translator.translate(ops.IfElse(op.where, op.arg, None))
    else:
        arg_formatted = translator.translate(op.arg)
    return f"count(DISTINCT {arg_formatted})"
