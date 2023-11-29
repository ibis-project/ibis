from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute


@execute.register(ops.StringLength)
def execute_string_length(op, arg):
    return arg.str.len().astype("int32")


@execute.register(ops.StringReplace)
def execute_string_replace(op, arg, pattern, replacement):
    return arg.str.replace(pattern, replacement)


@execute.register(ops.Strip)
def execute_string_strip(op, arg):
    return arg.str.strip()


@execute.register(ops.LStrip)
def execute_lstrip(op, arg):
    return arg.str.lstrip()


@execute.register(ops.RStrip)
def execute_rstrip(op, arg):
    return arg.str.rstrip()


@execute.register(ops.Reverse)
def execute_string_reverse(op, arg):
    return arg.str[::-1]


@execute.register(ops.Lowercase)
def execute_string_lower(op, arg):
    return arg.str.lower()


@execute.register(ops.Uppercase)
def execute_string_upper(op, arg):
    return arg.str.upper()
