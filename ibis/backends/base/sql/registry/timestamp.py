from __future__ import annotations

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util


def extract_field(sql_attr):
    def extract_field_formatter(translator, op):
        arg = translator.translate(op.args[0])

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return f"extract({arg}, '{sql_attr}')"

    return extract_field_formatter


def extract_epoch_seconds(t, op):
    return f"unix_timestamp({t.translate(op.arg)})"


def truncate(translator, op):
    base_unit_names = {
        "Y": "Y",
        "Q": "Q",
        "M": "MONTH",
        "W": "W",
        "D": "J",
        "h": "HH",
        "m": "MI",
    }
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = base_unit_names[unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(
            f"{unit!r} unit is not supported in timestamp truncate"
        )

    return f"trunc({arg_formatted}, '{unit}')"


def interval_from_integer(translator, op):
    # interval cannot be selected from impala
    arg = translator.translate(op.arg)
    return f"INTERVAL {arg} {op.dtype.resolution.upper()}"


def timestamp_op(func):
    def _formatter(translator, op):
        formatted_left = translator.translate(op.left)
        formatted_right = translator.translate(op.right)

        left_dtype = op.left.dtype
        if left_dtype.is_timestamp() or left_dtype.is_date():
            formatted_left = f"cast({formatted_left} as timestamp)"

        right_dtype = op.right.dtype
        if right_dtype.is_timestamp() or right_dtype.is_date():
            formatted_right = f"cast({formatted_right} as timestamp)"

        return f"{func}({formatted_left}, {formatted_right})"

    return _formatter


def timestamp_diff(translator, op):
    return "unix_timestamp({}) - unix_timestamp({})".format(
        translator.translate(op.left), translator.translate(op.right)
    )


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return f'from_unixtime({arg}, "yyyy-MM-dd HH:mm:ss")'


def timestamp_from_unix(translator, op):
    val, unit = op.args
    val = util.convert_unit(val, unit.short, "s").to_expr().cast("int32").op()
    arg = _from_unixtime(translator, val)
    return f"CAST({arg} AS timestamp)"


def day_of_week_index(t, op):
    return f"pmod(dayofweek({t.translate(op.arg)}) - 2, 7)"


def strftime(t, op):
    import sqlglot as sg

    hive_dialect = sg.dialects.hive.Hive
    if (time_mapping := getattr(hive_dialect, "TIME_MAPPING", None)) is None:
        time_mapping = hive_dialect.time_mapping
    reverse_hive_mapping = {v: k for k, v in time_mapping.items()}
    format_str = sg.time.format_time(op.format_str.value, reverse_hive_mapping)
    targ = t.translate(ops.Cast(op.arg, to=dt.string))
    return f"from_unixtime(unix_timestamp({targ}), {format_str!r})"


def day_of_week_name(t, op):
    return f"dayname({t.translate(op.arg)})"
