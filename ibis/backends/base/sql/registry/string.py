from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import helpers


def substring(translator, op):
    arg, start, length = op.args
    arg_formatted = translator.translate(arg)
    start_formatted = translator.translate(start)

    # Impala is 1-indexed
    if length is None or isinstance(length, ops.Literal):
        if lvalue := getattr(length, "value", None):
            return f"substr({arg_formatted}, {start_formatted} + 1, {lvalue})"
        else:
            return f"substr({arg_formatted}, {start_formatted} + 1)"
    else:
        length_formatted = translator.translate(length)
        return f"substr({arg_formatted}, {start_formatted} + 1, {length_formatted})"


def string_find(translator, op):
    arg_formatted = translator.translate(op.arg)
    substr_formatted = translator.translate(op.substr)

    if (start := op.start) is not None:
        if not isinstance(start, ops.Literal):
            start_fmt = translator.translate(start)
            return f"locate({substr_formatted}, {arg_formatted}, {start_fmt} + 1) - 1"
        elif sval := start.value:
            return f"locate({substr_formatted}, {arg_formatted}, {sval + 1}) - 1"
        else:
            raise ValueError(f"invalid `start` value: {sval}")
    else:
        return f"locate({substr_formatted}, {arg_formatted}) - 1"


def find_in_set(translator, op):
    arg_formatted = translator.translate(op.needle)
    str_formatted = ",".join([x.value for x in op.values])
    return f"find_in_set({arg_formatted}, '{str_formatted}') - 1"


def string_join(translator, op):
    arg, strings = op.args
    return helpers.format_call(translator, "concat_ws", arg, *strings)


def string_like(translator, op):
    arg = translator.translate(op.arg)
    pattern = translator.translate(op.pattern)
    return f"{arg} LIKE {pattern}"


def string_ilike(translator, op):
    arg = translator.translate(op.arg)
    pattern = translator.translate(op.pattern)
    return f"upper({arg}) LIKE upper({pattern})"


def extract_url_field(extract):
    if extract == "QUERY":

        def _op(translator, op):
            arg, key = op.args
            arg_formatted = translator.translate(arg)

            if key is None:
                return f"parse_url({arg_formatted}, '{extract}')"
            else:
                key_fmt = translator.translate(key)
                return f"parse_url({arg_formatted}, '{extract}', {key_fmt})"

    else:

        def _op(translator, op):
            arg_formatted = translator.translate(op.arg)
            return f"parse_url({arg_formatted}, '{extract}')"

    return _op


def startswith(translator, op):
    arg_formatted = translator.translate(op.arg)
    start_formatted = translator.translate(op.start)

    return f"{arg_formatted} like concat({start_formatted}, '%')"


def endswith(translator, op):
    arg_formatted = translator.translate(op.arg)
    end_formatted = translator.translate(op.end)

    return f"{arg_formatted} like concat('%', {end_formatted})"
