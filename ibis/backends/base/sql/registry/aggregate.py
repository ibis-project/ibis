from __future__ import annotations

import itertools

import ibis
import ibis.expr.operations as ops


def _reduction_format(translator, func_name, where, arg, *args):
    if where is not None:
        arg = ops.IfElse(where, arg, ibis.NA)

    return "{}({})".format(
        func_name,
        ", ".join(map(translator.translate, itertools.chain([arg], args))),
    )


def reduction(func_name):
    def formatter(translator, op):
        *args, where = op.args
        return _reduction_format(translator, func_name, where, *args)

    return formatter


def variance_like(func_name):
    func_names = {
        "sample": f"{func_name}_samp",
        "pop": f"{func_name}_pop",
    }

    def formatter(translator, op):
        return _reduction_format(translator, func_names[op.how], op.where, op.arg)

    return formatter


def count_distinct(translator, op):
    if op.where is not None:
        arg_formatted = translator.translate(ops.IfElse(op.where, op.arg, None))
    else:
        arg_formatted = translator.translate(op.arg)
    return f"count(DISTINCT {arg_formatted})"
