from __future__ import annotations

import calendar
import functools
import math
from functools import partial
from typing import TYPE_CHECKING, Any

import sqlglot as sg
from toolz import flip

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.datatypes import DuckDBType

if TYPE_CHECKING:
    from collections.abc import Mapping


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_val.register(dt.DataType)
def _datatype(t, **_):
    return DuckDBType.from_ibis(t)


@translate_val.register(ops.PhysicalTable)
def _val_physical_table(op, *, aliases, **kw):
    return f"{aliases.get(op, op.name)}.*"


@translate_val.register(ops.TableNode)
def _val_table_node(op, *, aliases, needs_alias=False, **_):
    return f"{aliases[op]}.*" if needs_alias else "*"


@translate_val.register(ops.TableColumn)
def _column(op, *, aliases, **_):
    table_name = (aliases or {}).get(op.table)
    return sg.column(op.name, table=table_name)


@translate_val.register(ops.Alias)
def _alias(op, render_aliases: bool = True, **kw):
    val = translate_val(op.arg, render_aliases=render_aliases, **kw)
    if render_aliases:
        return sg.alias(val, op.name)
    return val


### Literals


def _sql(obj):
    try:
        return obj.sql(dialect="duckdb")
    except AttributeError:
        return obj


def sg_literal(arg, is_string=True):
    return sg.exp.Literal(this=f"{arg}", is_string=is_string)


@translate_val.register(ops.Literal)
def _literal(op, **kw):
    value = op.value
    dtype = op.dtype
    sg_type = DuckDBType.from_ibis(dtype)

    if value is None and dtype.nullable:
        if dtype.is_null():
            return sg.exp.Null()
        return sg.cast(sg.exp.Null(), to=sg_type)
    elif dtype.is_boolean():
        return sg.exp.Boolean(this=value)
    elif dtype.is_inet():
        return sg.exp.Literal(this=str(value), is_string=True)
    elif dtype.is_string():
        return sg_literal(value)
    elif dtype.is_decimal():
        # TODO: make this a sqlglot expression
        precision = dtype.precision
        scale = dtype.scale
        if precision is None:
            precision = 38
        if scale is None:
            scale = 9
        if not 1 <= precision <= 38:
            raise NotImplementedError(
                f"Unsupported precision. Supported values: [1 : 38]. Current value: {precision!r}"
            )
        if math.isinf(value):
            return sg.exp.cast(
                expression=sg_literal(value),
                to=sg.exp.DataType.Type.FLOAT,
            )
        elif math.isnan(value):
            return sg.exp.cast(
                expression=sg_literal("NaN"),
                to=sg.exp.DataType.Type.FLOAT,
            )

        dtype = dt.Decimal(precision=precision, scale=scale, nullable=dtype.nullable)
        sg_expr = sg.cast(sg_literal(value, is_string=False), to=sg_type)
        return sg_expr
    elif dtype.is_numeric():
        if math.isinf(value):
            return sg.exp.cast(
                expression=sg_literal(value),
                to=sg.exp.DataType.Type.FLOAT,
            )
        elif math.isnan(value):
            return sg.exp.cast(
                expression=sg_literal("NaN"),
                to=sg.exp.DataType.Type.FLOAT,
            )
        return sg.cast(sg_literal(value, is_string=False), to=sg_type)
    elif dtype.is_interval():
        return _interval_format(op)
    elif dtype.is_timestamp():
        year = sg_literal(op.value.year, is_string=False)
        month = sg_literal(op.value.month, is_string=False)
        day = sg_literal(op.value.day, is_string=False)
        hour = sg_literal(op.value.hour, is_string=False)
        minute = sg_literal(op.value.minute, is_string=False)
        second = sg_literal(op.value.second, is_string=False)
        if op.value.microsecond:
            microsecond = sg_literal(op.value.microsecond / 1e6, is_string=False)
            second += microsecond
        if dtype.timezone is not None:
            timezone = sg_literal(dtype.timezone, is_string=True)
            return sg.func(
                "make_timestamptz", year, month, day, hour, minute, second, timezone
            )
        else:
            return sg.func("make_timestamp", year, month, day, hour, minute, second)
    elif dtype.is_date():
        year = sg_literal(op.value.year, is_string=False)
        month = sg_literal(op.value.month, is_string=False)
        day = sg_literal(op.value.day, is_string=False)
        return sg.exp.DateFromParts(year=year, month=month, day=day)
    elif dtype.is_array():
        value_type = dtype.value_type
        is_string = isinstance(value_type, dt.String)
        values = sg.exp.Array().from_arg_list(
            [
                # TODO: this cast makes for frustrating output
                # is there any better way to handle it?
                sg.cast(sg_literal(v, is_string=is_string), to=sg_type)
                for v in value
            ]
        )
        return values
    elif dtype.is_map():
        key_type = dtype.key_type
        value_type = dtype.value_type
        keys = sg.exp.Array().from_arg_list(
            [_literal(ops.Literal(k, dtype=key_type), **kw) for k in value.keys()]
        )
        values = sg.exp.Array().from_arg_list(
            [_literal(ops.Literal(v, dtype=value_type), **kw) for v in value.values()]
        )
        sg_expr = sg.exp.Map(keys=keys, values=values)
        return sg_expr
    elif dtype.is_struct():
        keys = [sg_literal(key) for key in value.keys()]
        values = [
            _literal(ops.Literal(v, dtype=subdtype), **kw)
            for subdtype, v in zip(dtype.types, value.values())
        ]
        slices = [sg.exp.Slice(this=k, expression=v) for k, v in zip(keys, values)]
        sg_expr = sg.exp.Struct.from_arg_list(slices)
        return sg_expr
    elif dtype.is_uuid():
        return sg.cast(sg_literal(value, is_string=True), to=sg_type)
    elif dtype.is_binary():
        bytestring = "".join(map("\\x{:02x}".format, value))
        lit = sg_literal(bytestring)
        return sg.cast(lit, to=sg_type)
    else:
        raise NotImplementedError(f"Unsupported type: {dtype!r}")


### Simple Ops

_simple_ops = {
    ops.Power: "pow",
    # Unary operations
    ops.IsNan: "isnan",
    ops.IsInf: "isinf",
    ops.Abs: "abs",
    ops.Ceil: "ceil",
    ops.Floor: "floor",
    ops.Exp: "exp",
    ops.Sqrt: "sqrt",
    ops.Ln: "ln",
    ops.Log2: "log2",
    ops.Log10: "log",
    ops.Acos: "acos",
    ops.Asin: "asin",
    ops.Atan: "atan",
    ops.Atan2: "atan2",
    ops.Cos: "cos",
    ops.Sin: "sin",
    ops.Tan: "tan",
    ops.Cot: "cot",
    ops.Pi: "pi",
    ops.RandomScalar: "random",
    ops.Sign: "sign",
    # Unary aggregates
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.Median: "median",
    ops.Mean: "avg",
    ops.Max: "max",
    ops.Min: "min",
    ops.ArgMin: "arg_min",
    ops.Mode: "mode",
    ops.ArgMax: "arg_max",
    ops.First: "first",
    ops.Last: "last",
    ops.Count: "count",
    # string operations
    ops.StringContains: "contains",
    ops.StringLength: "length",
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.Reverse: "reverse",
    ops.StringReplace: "replace",
    ops.StartsWith: "prefix",
    ops.EndsWith: "suffix",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.LStrip: "ltrim",
    ops.RStrip: "rtrim",
    ops.Strip: "trim",
    ops.StringAscii: "ascii",
    ops.StrRight: "right",
    # Other operations
    ops.Where: "if",
    ops.ArrayLength: "length",
    ops.Unnest: "unnest",
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.NullIf: "nullIf",
    ops.MapLength: "cardinality",
    ops.MapKeys: "map_keys",
    ops.MapValues: "map_values",
    ops.ArraySort: "list_sort",
    ops.ArrayContains: "list_contains",
    ops.FirstValue: "first_value",
    ops.LastValue: "last_value",
    ops.NTile: "ntile",
    ops.Hash: "hash",
}


def _agg(func_name):
    def formatter(op, **kw):
        return _aggregate(op, func_name, where=op.where, **kw)

    return formatter


for _op, _name in _simple_ops.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):
        translate_val.register(_op)(_agg(_name))
    else:

        @translate_val.register(_op)
        def _fmt(op, _name: str = _name, **kw):
            return sg.func(_name, *map(partial(translate_val, **kw), op.args))


del _fmt, _name, _op
### Bitwise Business

_bitwise_mapping = {
    ops.BitwiseLeftShift: sg.exp.BitwiseLeftShift,
    ops.BitwiseRightShift: sg.exp.BitwiseRightShift,
    ops.BitwiseAnd: sg.exp.BitwiseAnd,
    ops.BitwiseOr: sg.exp.BitwiseOr,
    ops.BitwiseXor: sg.exp.BitwiseXor,
}


@translate_val.register(ops.BitwiseLeftShift)
@translate_val.register(ops.BitwiseRightShift)
@translate_val.register(ops.BitwiseAnd)
@translate_val.register(ops.BitwiseOr)
@translate_val.register(ops.BitwiseXor)
def _bitwise_binary(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    sg_expr = _bitwise_mapping[type(op)]

    return sg_expr(this=left, expression=right)


@translate_val.register(ops.BitwiseNot)
def _bitwise_not(op, **kw):
    value = translate_val(op.arg, **kw)

    return sg.exp.BitwiseNot(this=value)


### Mathematical Calisthenics


@translate_val.register(ops.E)
def _euler(op, **kw):
    return sg.func("exp", 1)


@translate_val.register(ops.Log)
def _generic_log(op, **kw):
    arg, base = op.args
    arg = translate_val(arg, **kw)
    if base is not None:
        base = translate_val(base, **kw)
        return sg.func("ln", arg) / sg.func("ln", base)
    return sg.func("ln", arg)


@translate_val.register(ops.Clip)
def _clip(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (upper := op.upper) is not None:
        arg = sg.exp.Least.from_arg_list([translate_val(upper, **kw), arg])

    if (lower := op.lower) is not None:
        arg = sg.exp.Greatest.from_arg_list([translate_val(lower, **kw), arg])

    return arg


@translate_val.register(ops.FloorDivide)
def _floor_divide(op, **kw):
    new_op = ops.Floor(ops.Divide(op.left, op.right))
    return translate_val(new_op, **kw)


@translate_val.register(ops.Round)
def _round(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (digits := op.digits) is not None:
        return sg.exp.Round(this=arg, decimals=translate_val(digits, **kw))
    return sg.exp.Round(this=arg)


### Dtype Dysmorphia


_interval_cast_suffixes = {
    "s": "Second",
    "m": "Minute",
    "h": "Hour",
    "D": "Day",
    "W": "Week",
    "M": "Month",
    "Q": "Quarter",
    "Y": "Year",
}


@translate_val.register(ops.Cast)
def _cast(op, **kw):
    arg = translate_val(op.arg, **kw)

    if isinstance(op.to, dt.Interval):
        suffix = _interval_cast_suffixes[op.to.unit.short]
        if isinstance(op.arg, ops.TableColumn):
            return (
                f"INTERVAL (i) {suffix} FROM (SELECT {arg.name} FROM {arg.table}) t(i)"
            )

        else:
            return sg.exp.Interval(this=arg, unit=suffix)
    elif isinstance(op.to, dt.Timestamp) and isinstance(op.arg.dtype, dt.Integer):
        return sg.func("to_timestamp", arg)
    elif isinstance(op.to, dt.Timestamp) and op.to.timezone is not None:
        timezone = sg.exp.Literal(this=op.to.timezone, is_string=True)
        return sg.func("timezone", timezone, arg)

    to = translate_val(op.to, **kw)
    return sg.cast(expression=arg, to=to)


@translate_val.register(ops.TryCast)
def _try_cast(op, **kw):
    return sg.exp.TryCast(
        this=translate_val(op.arg, **kw),
        to=DuckDBType.to_string(op.to),
        dialect="duckdb",
    )


@translate_val.register(ops.TypeOf)
def _type_of(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.func("typeof", arg)


### Comparator Conundrums


@translate_val.register(ops.Between)
def _between(op, **kw):
    arg = translate_val(op.arg, **kw)
    lower_bound = translate_val(op.lower_bound, **kw)
    upper_bound = translate_val(op.upper_bound, **kw)
    return sg.exp.Between(this=arg, low=lower_bound, high=upper_bound)


@translate_val.register(ops.Negate)
def _negate(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.exp.Neg(this=arg)


@translate_val.register(ops.Not)
def _not(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.exp.Not(this=arg)


def _apply_agg_filter(expr, *, where, **kw):
    if where is not None:
        return sg.exp.Filter(
            this=expr, expression=sg.exp.Where(this=translate_val(where, **kw))
        )
    return expr


def _aggregate(op, func, *, where, **kw):
    args = [
        translate_val(arg, **kw)
        for argname, arg in zip(op.argnames, op.args)
        if argname not in ("where", "how")
    ]
    agg = sg.func(func, *args)
    return _apply_agg_filter(agg, where=op.where, **kw)


@translate_val.register(ops.Any)
def _any(op, **kw):
    arg = translate_val(op.arg, **kw)
    any_expr = sg.func("bool_or", arg)
    return _apply_agg_filter(any_expr, where=op.where, **kw)


@translate_val.register(ops.All)
def _all(op, **kw):
    arg = translate_val(op.arg, **kw)
    all_expr = sg.func("bool_and", arg)
    return _apply_agg_filter(all_expr, where=op.where, **kw)


@translate_val.register(ops.NotAny)
def _not_any(op, **kw):
    return translate_val(ops.All(ops.Not(op.arg), where=op.where), **kw)


@translate_val.register(ops.NotAll)
def _not_all(op, **kw):
    return translate_val(ops.Any(ops.Not(op.arg), where=op.where), **kw)


### Timey McTimeFace


@translate_val.register(ops.Date)
def _to_date(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.exp.Date(this=arg)


@translate_val.register(ops.DateFromYMD)
def _date_from_ymd(op, **kw):
    y = translate_val(op.year, **kw)
    m = translate_val(op.month, **kw)
    d = translate_val(op.day, **kw)
    return sg.exp.DateFromParts(year=y, month=m, day=d)


@translate_val.register(ops.Time)
def _time(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.cast(expression=arg, to=sg.exp.DataType.Type.TIME)


@translate_val.register(ops.TimeFromHMS)
def _time_from_hms(op, **kw):
    hours = translate_val(op.hours, **kw)
    minutes = translate_val(op.minutes, **kw)
    seconds = translate_val(op.seconds, **kw)
    return sg.func("make_time", hours, minutes, seconds)


@translate_val.register(ops.TimestampNow)
def _timestamp_now(op, **kw):
    """DuckDB current timestamp defaults to timestamp + tz."""
    return sg.cast(expression=sg.func("current_timestamp"), to="TIMESTAMP")


@translate_val.register(ops.TimestampFromUNIX)
def _timestamp_from_unix(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (unit := op.unit.short) in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    return sg.exp.UnixToTime(this=arg)


@translate_val.register(ops.TimestampFromYMDHMS)
def _timestamp_from_ymdhms(op, **kw):
    year = translate_val(op.year, **kw)
    month = translate_val(op.month, **kw)
    day = translate_val(op.day, **kw)
    hour = translate_val(op.hours, **kw)
    minute = translate_val(op.minutes, **kw)
    second = translate_val(op.seconds, **kw)

    if (timezone := op.dtype.timezone) is not None:
        return f"make_timestamptz({year}, {month}, {day}, {hour}, {minute}, {second}, '{timezone}')"
    else:
        return f"make_timestamp({year}, {month}, {day}, {hour}, {minute}, {second})"


@translate_val.register(ops.Strftime)
def _strftime(op, **kw):
    if not isinstance(op.format_str, ops.Literal):
        raise com.UnsupportedOperationError(
            f"DuckDB format_str must be a literal `str`; got {type(op.format_str)}"
        )
    arg = translate_val(op.arg, **kw)
    format_str = translate_val(op.format_str, **kw)
    return sg.func("strftime", arg, format_str)


@translate_val.register(ops.StringToTimestamp)
def _string_to_timestamp(op, **kw):
    arg = translate_val(op.arg, **kw)
    format_str = translate_val(op.format_str, **kw)
    return sg.func("strptime", arg, format_str)


@translate_val.register(ops.ExtractEpochSeconds)
def _extract_epoch_seconds(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.func(
        "epoch",
        sg.exp.cast(
            expression=arg,
            to=sg.exp.DataType.Type.TIMESTAMP,
        ),
    )


_extract_mapping = {
    ops.ExtractYear: "year",
    ops.ExtractMonth: "month",
    ops.ExtractDay: "day",
    ops.ExtractDayOfYear: "dayofyear",
    ops.ExtractQuarter: "quarter",
    ops.ExtractWeekOfYear: "week",
    ops.ExtractHour: "hour",
    ops.ExtractMinute: "minute",
    ops.ExtractSecond: "second",
}


@translate_val.register(ops.ExtractYear)
@translate_val.register(ops.ExtractMonth)
@translate_val.register(ops.ExtractDay)
@translate_val.register(ops.ExtractDayOfYear)
@translate_val.register(ops.ExtractQuarter)
@translate_val.register(ops.ExtractWeekOfYear)
@translate_val.register(ops.ExtractHour)
@translate_val.register(ops.ExtractMinute)
@translate_val.register(ops.ExtractSecond)
def _extract_time(op, **kw):
    part = _extract_mapping[type(op)]
    timestamp = translate_val(op.arg, **kw)
    return sg.func("extract", sg.exp.Literal(this=part, is_string=True), timestamp)


# DuckDB extracts subminute microseconds and milliseconds
# so we have to finesse it a little bit
@translate_val.register(ops.ExtractMicrosecond)
def _extract_microsecond(op, **kw):
    arg = translate_val(op.arg, **kw)

    return sg.exp.Mod(
        this=sg.func(
            "extract",
            sg.exp.Literal(this="us", is_string=True),
            arg,
        ),
        expression=sg.exp.Literal(this="1000000", is_string=False),
    )


@translate_val.register(ops.ExtractMillisecond)
def _extract_microsecond(op, **kw):
    arg = translate_val(op.arg, **kw)

    return sg.exp.Mod(
        this=sg.func(
            "extract",
            sg.exp.Literal(this="ms", is_string=True),
            arg,
        ),
        expression=sg.exp.Literal(this="1000", is_string=False),
    )


@translate_val.register(ops.DateTruncate)
@translate_val.register(ops.TimestampTruncate)
@translate_val.register(ops.TimeTruncate)
def _truncate(op, **kw):
    unit_mapping = {
        "Y": "year",
        "M": "month",
        "W": "week",
        "D": "day",
        "h": "hour",
        "m": "minute",
        "s": "second",
        "ms": "ms",
        "us": "us",
    }

    unit = op.unit.short
    arg = translate_val(op.arg, **kw)
    try:
        duckunit = unit_mapping[unit]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

    return f"date_trunc('{duckunit}', {arg})"


@translate_val.register(ops.DayOfWeekIndex)
def _day_of_week_index(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"(dayofweek({arg}) + 6) % 7"


@translate_val.register(ops.DayOfWeekName)
def day_of_week_name(op, **kw):
    # day of week number is 0-indexed
    # Sunday == 0
    # Saturday == 6
    arg = op.arg
    nullable = arg.dtype.nullable
    empty_string = ops.Literal("", dtype=dt.String(nullable=nullable))
    weekdays = range(7)
    return translate_val(
        ops.NullIf(
            ops.SimpleCase(
                base=ops.DayOfWeekIndex(arg),
                cases=[
                    ops.Literal(day, dtype=dt.Int8(nullable=nullable))
                    for day in weekdays
                ],
                results=[
                    ops.Literal(
                        calendar.day_name[day],
                        dtype=dt.String(nullable=nullable),
                    )
                    for day in weekdays
                ],
                default=empty_string,
            ),
            empty_string,
        ),
        **kw,
    )


### Interval Marginalia


_interval_mapping = {
    ops.IntervalAdd: sg.exp.Add,
    ops.IntervalSubtract: sg.exp.Sub,
    ops.IntervalMultiply: sg.exp.Mul,
}


@translate_val.register(ops.IntervalAdd)
@translate_val.register(ops.IntervalSubtract)
@translate_val.register(ops.IntervalMultiply)
def _interval_binary(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    sg_expr = _interval_mapping[type(op)]

    return sg_expr(this=left, expression=right)


def _interval_format(op):
    dtype = op.dtype
    if dtype.unit.short == "ns":
        raise com.UnsupportedOperationError(
            "Duckdb doesn't support nanosecond interval resolutions"
        )

    return sg.exp.Interval(
        this=sg.exp.Literal(this=op.value, is_string=False),
        unit=dtype.resolution.upper(),
    )


@translate_val.register(ops.IntervalFromInteger)
def _interval_from_integer(op, **kw):
    dtype = op.dtype
    if dtype.unit.short == "ns":
        raise com.UnsupportedOperationError(
            "Duckdb doesn't support nanosecond interval resolutions"
        )

    arg = translate_val(op.arg, **kw)
    if op.dtype.resolution == "week":
        return sg.func("to_days", arg * 7)
    return sg.func(f"to_{op.dtype.resolution}s", arg)


### String Instruments


@translate_val.register(ops.Substring)
def _substring(op, **kw):
    arg = translate_val(op.arg, **kw)
    start = translate_val(op.start, **kw)
    if op.length is not None:
        length = translate_val(op.length, **kw)
    else:
        length = None

    if_pos = sg.exp.Substring(this=arg, start=start + 1, length=length)
    if_neg = sg.exp.Substring(this=arg, start=start, length=length)

    sg_expr = sg.exp.If(
        this=sg.exp.GTE(
            this=start, expression=sg.exp.Literal(this="0", is_string=False)
        ),
        true=if_pos,
        false=if_neg,
    )

    return sg_expr


@translate_val.register(ops.StringFind)
def _string_find(op, **kw):
    if op.end is not None:
        raise com.UnsupportedOperationError("String find doesn't support end argument")

    arg = translate_val(op.arg, **kw)
    substr = translate_val(op.substr, **kw)

    return sg.func("instr", arg, substr) - 1


@translate_val.register(ops.RegexSearch)
def _regex_search(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return sg.func(
        "regexp_matches", arg, pattern, sg.exp.Literal(this="s", is_string=True)
    )


@translate_val.register(ops.RegexReplace)
def _regex_replace(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    replacement = translate_val(op.replacement, **kw)
    return sg.func(
        "regexp_replace",
        arg,
        pattern,
        replacement,
        sg.exp.Literal(this="g", is_string=True),
        dialect="duckdb",
    )


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    group = translate_val(op.index, **kw)
    return sg.func("regexp_extract", arg, pattern, group)


@translate_val.register(ops.Levenshtein)
def _levenshtein(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return sg.func("levenshtein", left, right)


@translate_val.register(ops.StringSplit)
def _string_split(op, **kw):
    arg = translate_val(op.arg, **kw)
    delimiter = translate_val(op.delimiter, **kw)
    return sg.exp.Split(this=arg, expression=delimiter)


@translate_val.register(ops.StringJoin)
def _string_join(op, **kw):
    elements = list(map(partial(translate_val, **kw), op.arg))
    sep = translate_val(op.sep, **kw)
    return sg.func(
        "list_aggr", sg.exp.Array(expressions=elements), sg_literal("string_agg"), sep
    )


@translate_val.register(ops.StringConcat)
def _string_concat(op, **kw):
    arg = map(partial(translate_val, **kw), op.arg)
    return sg.exp.Concat(expressions=list(arg))


@translate_val.register(ops.StringSQLLike)
def _string_like(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return sg.exp.Like(this=arg, expression=pattern)


@translate_val.register(ops.StringSQLILike)
def _string_ilike(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return sg.exp.ILike(this=arg, expression=pattern)


@translate_val.register(ops.Capitalize)
def _string_capitalize(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.exp.Concat(
        expressions=[
            sg.func("upper", sg.func("substr", arg, 1, 1)),
            sg.func("lower", sg.func("substr", arg, 2)),
        ]
    )


### NULL PLAYER CHARACTER
@translate_val.register(ops.IsNull)
def _is_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return arg.is_(sg.exp.null())


@translate_val.register(ops.NotNull)
def _is_not_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return arg.is_(sg.not_(sg.exp.null()))


@translate_val.register(ops.IfNull)
def _if_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    ifnull = translate_val(op.ifnull_expr, **kw)
    return sg.func("ifnull", arg, ifnull)


@translate_val.register(ops.NullIfZero)
def _null_if_zero(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.func("nullif", arg, 0)


@translate_val.register(ops.ZeroIfNull)
def _zero_if_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.func("ifnull", arg, 0)


### Definitely Not Tensors


@translate_val.register(ops.ArrayDistinct)
def _array_sort(op, **kw):
    arg = translate_val(op.arg, **kw)

    sg_expr = sg.exp.If(
        this=arg.is_(sg.exp.Null()),
        true=sg.exp.Null(),
        false=sg.func("list_distinct", arg)
        + sg.exp.If(
            this=sg.func("list_count", arg) < sg.func("array_length", arg),
            true=sg.func("list_value", sg.exp.Null()),
            false=sg.func("list_value"),
        ),
    )
    return sg_expr


@translate_val.register(ops.ArrayIndex)
def _array_index_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    index = translate_val(op.index, **kw)
    correct_idx = f"if({index} >= 0, {index} + 1, {index})"
    return f"array_extract({arg}, {correct_idx})"


@translate_val.register(ops.InValues)
def _in_values(op, **kw):
    if not op.options:
        return sg.exp.FALSE

    value = translate_val(op.value, **kw)
    return sg.exp.In(
        this=value,
        expressions=[translate_val(opt, **kw) for opt in op.options],
    )


@translate_val.register(ops.InColumn)
def _in_column(op, **kw):
    value = translate_val(op.value, **kw)
    options = translate_val(ops.TableArrayView(op.options.to_expr().as_table()), **kw)
    # TODO: fix?
    # if not isinstance(options, sa.sql.Selectable):
    #     options = sg.select(options)
    return value.isin(options)


@translate_val.register(ops.ArrayCollect)
def _array_collect(op, **kw):
    agg = sg.func("list", translate_val(op.arg, **kw))
    return _apply_agg_filter(agg, where=op.where, **kw)


@translate_val.register(ops.ArrayConcat)
def _array_concat(op, **kw):
    sg_expr = sg.func(
        "flatten",
        sg.func(
            "list_value",
            *(translate_val(arg, **kw) for arg in op.arg),
        ),
        dialect="duckdb",
    )
    return sg_expr


@translate_val.register(ops.ArrayRepeat)
def _array_repeat_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    sg_expr = sg.func(
        "flatten",
        sg.select(
            sg.func("array", sg.select(arg).from_(sg.func("range", times)))
        ).subquery(),
    )
    return sg_expr


def _neg_idx_to_pos(array, idx):
    arg_length = sg.func("len", array)
    return sg.exp.If(
        this=sg.exp.LT(this=idx, expression=sg_literal(0, is_string=False)),
        # Need to have the greatest here to handle the case where
        # abs(neg_index) > arg_length
        # e.g. where the magnitude of the negative index is greater than the
        # length of the array
        # You cannot index a[:-3] if a = [1, 2]
        true=arg_length + sg.func("greatest", idx, -1 * arg_length),
        false=idx,
    )


@translate_val.register(ops.ArraySlice)
def _array_slice_op(op, **kw):
    arg = translate_val(op.arg, **kw)

    arg_length = sg.func("len", arg)

    if (start := op.start) is None:
        start = sg_literal(0, is_string=False)
    else:
        start = translate_val(op.start, **kw)
        start = sg.func("least", arg_length, _neg_idx_to_pos(arg, start))

    if (stop := op.stop) is None:
        stop = sg.exp.Null()
    else:
        stop = _neg_idx_to_pos(arg, translate_val(stop, **kw))

    return sg.func("list_slice", arg, start + 1, stop)


@translate_val.register(ops.ArrayStringJoin)
def _array_string_join(op, **kw):
    arg = translate_val(op.arg, **kw)
    sep = translate_val(op.sep, **kw)
    return f"list_aggregate({arg}, 'string_agg', {sep})"


@translate_val.register(ops.ArrayMap)
def _array_map(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    lamduh = sg.exp.Lambda(
        this=result,
        expressions=[sg.to_identifier(f"{op.parameter}", quoted=False)],
    )
    sg_expr = sg.func("list_transform", arg, lamduh)
    return sg_expr


@translate_val.register(ops.ArrayFilter)
def _array_filter(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    lamduh = sg.exp.Lambda(
        this=result,
        expressions=[sg.exp.Identifier(this=f"{op.parameter}", quoted=False)],
    )
    sg_expr = sg.func("list_filter", arg, lamduh)
    return sg_expr


@translate_val.register(ops.ArrayIntersect)
def _array_intersect(op, **kw):
    return translate_val(
        ops.ArrayFilter(op.left, func=lambda x: ops.ArrayContains(op.right, x)), **kw
    )


@translate_val.register(ops.ArrayPosition)
def _array_position(op, **kw):
    arg = translate_val(op.arg, **kw)
    el = translate_val(op.other, **kw)
    return sg.func("list_indexof", arg, el) - 1


@translate_val.register(ops.ArrayRemove)
def _array_remove(op, **kw):
    return translate_val(ops.ArrayFilter(op.arg, flip(ops.NotEquals, op.other)), **kw)


@translate_val.register(ops.ArrayUnion)
def _array_union(op, **kw):
    return translate_val(ops.ArrayDistinct(ops.ArrayConcat((op.left, op.right))), **kw)


@translate_val.register(ops.ArrayZip)
def _array_zip(op: ops.ArrayZip, **kw: Any) -> str:
    i = sg.to_identifier("i", quoted=False)
    args = [translate_val(arg, **kw) for arg in op.arg]
    result = sg.exp.Struct(
        expressions=[
            sg.exp.Slice(
                this=sg_literal(name),
                expression=sg.func("list_extract", arg, i),
            )
            for name, arg in zip(op.dtype.value_type.names, args)
        ]
    )
    lamduh = sg.exp.Lambda(this=result, expressions=[i])
    sg_expr = sg.func(
        "list_transform",
        sg.func(
            "range",
            sg_literal(1, is_string=False),
            # DuckDB Range is not inclusive of upper bound
            sg.func("greatest", *[sg.func("len", arg) for arg in args]) + 1,
        ),
        lamduh,
        dialect="duckdb",
    )

    return sg_expr


### Counting


@translate_val.register(ops.CountDistinct)
def _count_distinct(op, **kw):
    arg = translate_val(op.arg, **kw)
    count_expr = sg.exp.Count(this=sg.exp.Distinct(expressions=[arg]))
    return _apply_agg_filter(count_expr, where=op.where, **kw)


@translate_val.register(ops.CountDistinctStar)
def _count_distinct_star(op, **kw):
    # use a tuple because duckdb doesn't accept COUNT(DISTINCT a, b, c, ...)
    #
    # this turns the expression into COUNT(DISTINCT (a, b, c, ...))
    row = sg.exp.Tuple(expressions=list(map(sg.column, op.arg.schema.keys())))
    expr = sg.exp.Count(this=sg.exp.Distinct(expressions=[row]))
    return _apply_agg_filter(expr, where=op.where, **kw)


@translate_val.register(ops.CountStar)
def _count_star(op, **kw):
    sql = sg.exp.Count(this=sg.exp.Star())
    return _apply_agg_filter(sql, where=op.where, **kw)


@translate_val.register(ops.Sum)
def _sum(op, **kw):
    arg = translate_val(
        ops.Cast(arg, to=op.dtype) if (arg := op.arg).dtype.is_boolean() else arg, **kw
    )
    return _apply_agg_filter(sg.exp.Sum(this=arg), where=op.where, **kw)


@translate_val.register(ops.NthValue)
def _nth_value(op, **kw):
    arg = translate_val(op.arg, **kw)
    nth = translate_val(op.nth, **kw)
    return sg.func("nth_value", arg, nth + 1)


@translate_val.register(ops.Repeat)
def _repeat(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    return sg.func("repeat", arg, times)


### Stats


@translate_val.register(ops.Quantile)
@translate_val.register(ops.MultiQuantile)
def _quantile(op, **kw):
    arg = translate_val(op.arg, **kw)
    quantile = translate_val(op.quantile, **kw)
    sg_expr = sg.func("quantile_cont", arg, quantile)
    return _apply_agg_filter(sg_expr, where=op.where, **kw)


@translate_val.register(ops.Correlation)
def _corr(op, **kw):
    if op.how == "sample":
        raise com.UnsupportedOperationError(
            "DuckDB only implements `pop` correlation coefficient"
        )

    left = translate_val(op.left, **kw)
    if (left_type := op.left.dtype).is_boolean():
        left = sg.cast(
            expression=left,
            to=DuckDBType.from_ibis(dt.Int32(nullable=left_type.nullable)),
        )

    right = translate_val(op.right, **kw)
    if (right_type := op.right.dtype).is_boolean():
        right = sg.cast(
            expression=right,
            to=DuckDBType.from_ibis(dt.Int32(nullable=right_type.nullable)),
        )

    sg_func = sg.func("corr", left, right)
    return _apply_agg_filter(sg_func, where=op.where, **kw)


@translate_val.register(ops.Covariance)
def _covariance(op, **kw):
    _how = {"sample": "samp", "pop": "pop"}

    left = translate_val(op.left, **kw)
    if (left_type := op.left.dtype).is_boolean():
        left = sg.cast(
            expression=left,
            to=DuckDBType.from_ibis(dt.Int32(nullable=left_type.nullable)),
        )

    right = translate_val(op.right, **kw)
    if (right_type := op.right.dtype).is_boolean():
        right = sg.cast(
            expression=right,
            to=DuckDBType.from_ibis(dt.Int32(nullable=right_type.nullable)),
        )

    sg_func = sg.func(f"covar_{_how[op.how]}", left, right)
    return _apply_agg_filter(sg_func, where=op.where, **kw)


@translate_val.register(ops.Variance)
@translate_val.register(ops.StandardDev)
def _variance(op, **kw):
    _how = {"sample": "samp", "pop": "pop"}
    _func = {ops.Variance: "var", ops.StandardDev: "stddev"}

    arg = op.arg
    if (arg_dtype := arg.dtype).is_boolean():
        arg = ops.Cast(arg, to=dt.Int32(nullable=arg_dtype))

    arg = translate_val(arg, **kw)

    sg_func = sg.func(f"{_func[type(op)]}_{_how[op.how]}", arg)
    return _apply_agg_filter(sg_func, where=op.where, **kw)


@translate_val.register(ops.Arbitrary)
def _arbitrary(op, **kw):
    if op.how == "heavy":
        raise com.UnsupportedOperationError("how='heavy' not supported in the backend")
    functions = {
        "first": "first",
        "last": "last",
    }
    return _aggregate(op, functions[op.how], where=op.where, **kw)


@translate_val.register(ops.FindInSet)
def _index_of(op: ops.FindInSet, **kw):
    needle = translate_val(op.needle, **kw)
    args = sg.exp.Array(expressions=list(map(partial(translate_val, **kw), op.values)))
    return sg.func("list_indexof", args, needle) - 1


@translate_val.register(tuple)
def _node_list(op, **kw):
    return sg.exp.Tuple(expressions=list(map(partial(translate_val, **kw), op)))


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, **kw):
    case = sg.exp.Case()

    if (base := getattr(op, "base", None)) is not None:
        case = sg.exp.Case(this=translate_val(base, **kw))

    for when, then in zip(op.cases, op.results):
        case = case.when(
            condition=translate_val(when, **kw),
            then=translate_val(then, **kw),
        )

    if (default := op.default) is not None:
        case = case.else_(condition=translate_val(default, **kw))

    return case


@translate_val.register(ops.TableArrayView)
def _table_array_view(op, *, cache, **kw):
    table = op.table
    try:
        return cache[table]
    except KeyError:
        from ibis.backends.duckdb.compiler.relations import translate_rel

        # ignore the top level table, so that we can compile its dependencies
        (leaf,) = an.find_immediate_parent_tables(table, keep_input=False)
        res = translate_rel(table, table=cache[leaf], cache=cache, **kw)
        return res.subquery()


# TODO
@translate_val.register(ops.ExistsSubquery)
@translate_val.register(ops.NotExistsSubquery)
def _exists_subquery(op, **kw):
    from ibis.backends.duckdb.compiler.relations import translate_rel

    if "table" not in kw:
        kw["table"] = translate_rel(op.foreign_table.table, **kw)
    foreign_table = translate_rel(op.foreign_table, **kw)
    predicates = translate_val(op.predicates, **kw)
    subq = sg.select(1).from_(foreign_table).where(sg.condition(predicates))
    prefix = "NOT " * isinstance(op, ops.NotExistsSubquery)
    return f"{prefix}EXISTS ({subq})"


@translate_val.register(ops.GroupConcat)
def _group_concat(op, **kw):
    arg = translate_val(op.arg, **kw)
    sep = translate_val(op.sep, **kw)

    concat = sg.func("string_agg", arg, sep)
    return _apply_agg_filter(concat, where=op.where, **kw)


@translate_val.register(ops.ArrayColumn)
def _array_column(op, **kw):
    return sg.exp.Array.from_arg_list([translate_val(col, **kw) for col in op.cols])


@translate_val.register(ops.StructColumn)
def _struct_column(op, **kw):
    return sg.exp.Struct(
        expressions=[
            sg.exp.Slice(
                this=sg.exp.Literal(this=name, is_string=True),
                expression=translate_val(value, **kw),
            )
            for name, value in zip(op.names, op.values)
        ]
    )


@translate_val.register(ops.StructField)
def _struct_field(op, **kw):
    arg = translate_val(op.arg, **kw)
    return sg.exp.StructExtract(
        this=arg, expression=sg.exp.Literal(this=op.field, is_string=True)
    )


@translate_val.register(ops.ScalarParameter)
def _scalar_param(op, params: Mapping[ops.Node, Any], **kw):
    raw_value = params[op]
    dtype = op.dtype
    if isinstance(dtype, dt.Struct):
        literal = ibis.struct(raw_value, type=dtype)
    elif isinstance(dtype, dt.Map):
        literal = ibis.map(raw_value)
    else:
        literal = ibis.literal(raw_value, type=dtype)
    return translate_val(literal.op(), **kw)


@translate_val.register(ops.IdenticalTo)
def _identical_to(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return sg.exp.NullSafeEQ(this=left, expression=right)


@translate_val.register(ops.Greatest)
@translate_val.register(ops.Least)
@translate_val.register(ops.Coalesce)
def _vararg_func(op, **kw):
    return sg.func(
        f"{op.__class__.__name__.lower()}",
        *map(partial(translate_val, **kw), op.arg),
        dialect="duckdb",
    )


@translate_val.register(ops.Map)
def _map(op, **kw):
    keys = translate_val(op.keys, **kw)
    values = translate_val(op.values, **kw)
    sg_expr = sg.exp.Map(keys=keys, values=values)
    return sg_expr


@translate_val.register(ops.MapGet)
def _map_get(op, **kw):
    arg = translate_val(op.arg, **kw)
    key = translate_val(op.key, **kw)
    default = translate_val(op.default, **kw)
    sg_expr = sg.func(
        "ifnull",
        sg.func("list_extract", sg.func("element_at", arg, key), 1),
        default,
        dialect="duckdb",
    )
    return sg_expr


@translate_val.register(ops.MapContains)
def _map_contains(op, **kw):
    arg = translate_val(op.arg, **kw)
    key = translate_val(op.key, **kw)
    sg_expr = sg.exp.NEQ(
        this=sg.func(
            "array_length",
            sg.func(
                "element_at",
                arg,
                key,
            ),
        ),
        expression=sg.exp.Literal(this="0", is_string=False),
    )
    return sg_expr


def _is_map_literal(op):
    return isinstance(op, ops.Literal) or (
        isinstance(op, ops.Map)
        and isinstance(op.keys, ops.Literal)
        and isinstance(op.values, ops.Literal)
    )


@translate_val.register(ops.MapMerge)
def _map_merge(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return sg.func("map_concat", left, right)


def _binary_infix(sg_expr: sg.exp._Expression):
    def formatter(op, **kw):
        left = translate_val(op.left, **kw)
        right = translate_val(op.right, **kw)

        return sg_expr(
            this=sg.exp.Paren(this=left),
            expression=sg.exp.Paren(this=right),
        )

    return formatter


_binary_infix_ops = {
    # Binary operations
    ops.Add: sg.exp.Add,
    ops.Subtract: sg.exp.Sub,
    ops.Multiply: sg.exp.Mul,
    ops.Divide: sg.exp.Div,
    ops.Modulus: sg.exp.Mod,
    # Comparisons
    ops.GreaterEqual: sg.exp.GTE,
    ops.Greater: sg.exp.GT,
    ops.LessEqual: sg.exp.LTE,
    ops.Less: sg.exp.LT,
    ops.Equals: sg.exp.EQ,
    ops.NotEquals: sg.exp.NEQ,
    # Boolean comparisons
    ops.And: sg.exp.And,
    ops.Or: sg.exp.Or,
    ops.DateAdd: sg.exp.Add,
    ops.DateSub: sg.exp.Sub,
    ops.DateDiff: sg.exp.Sub,
    ops.TimestampAdd: sg.exp.Add,
    ops.TimestampSub: sg.exp.Sub,
    ops.TimestampDiff: sg.exp.Sub,
}


for _op, _sym in _binary_infix_ops.items():
    translate_val.register(_op)(_binary_infix(_sym))

del _op, _sym


_bit_agg = {
    ops.BitOr: "bit_or",
    ops.BitAnd: "bit_and",
    ops.BitXor: "bit_xor",
}


@translate_val.register(ops.BitAnd)
@translate_val.register(ops.BitOr)
@translate_val.register(ops.BitXor)
def _bitor(op, **kw):
    arg = translate_val(op.arg, **kw)
    bit_expr = sg.func(_bit_agg[type(op)], arg)
    return _apply_agg_filter(bit_expr, where=op.where, **kw)


@translate_val.register(ops.Xor)
def _xor(op, **kw):
    # https://github.com/tobymao/sqlglot/issues/2238
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return sg.exp.And(
        this=sg.exp.Paren(this=sg.exp.Or(this=left, expression=right)),
        expression=sg.exp.Paren(
            this=sg.exp.Not(this=sg.exp.And(this=left, expression=right))
        ),
    )


### Ordering


@translate_val.register(ops.RowNumber)
def _row_number(_, **kw):
    return sg.exp.RowNumber()


@translate_val.register(ops.DenseRank)
def _dense_rank(_, **kw):
    return sg.func("dense_rank")


@translate_val.register(ops.MinRank)
def _rank(_, **kw):
    return sg.func("rank")


@translate_val.register(ops.PercentRank)
def _percent_rank(_, **kw):
    return sg.func("percent_rank")


@translate_val.register(ops.CumeDist)
def _cume_dist(_, **kw):
    return sg.func("percent_rank")


@translate_val.register
def _sort_key(op: ops.SortKey, **kw):
    arg = translate_val(op.expr, **kw)
    direction = "ASC" if op.ascending else "DESC"
    return f"{_sql(arg)} {direction}"


### Window functions

_cumulative_to_reduction = {
    ops.CumulativeSum: ops.Sum,
    ops.CumulativeMin: ops.Min,
    ops.CumulativeMax: ops.Max,
    ops.CumulativeMean: ops.Mean,
    ops.CumulativeAny: ops.Any,
    ops.CumulativeAll: ops.All,
}


def cumulative_to_window(func, frame):
    klass = _cumulative_to_reduction[type(func)]
    new_op = klass(*func.args)
    new_frame = frame.copy(start=None, end=0)
    new_expr = an.windowize_function(new_op.to_expr(), frame=new_frame)
    return new_expr.op()


def format_window_boundary(boundary, **kw):
    value = translate_val(boundary.value, **kw)
    if boundary.preceding:
        return f"{value} PRECEDING"
    else:
        return f"{value} FOLLOWING"


# TODO
def format_window_frame(func, frame, **kw):
    components = []

    if frame.how == "rows" and frame.max_lookback is not None:
        raise NotImplementedError(
            "Rows with max lookback is not implemented for the Duckdb backend."
        )

    if frame.group_by:
        partition_args = ", ".join(
            map(_sql, map(partial(translate_val, **kw), frame.group_by))
        )
        components.append(f"PARTITION BY {partition_args}")

    if frame.order_by:
        order_args = ", ".join(
            map(_sql, map(partial(translate_val, **kw), frame.order_by))
        )
        components.append(f"ORDER BY {order_args}")

    frame_clause_not_allowed = (
        ops.Lag,
        ops.Lead,
        ops.DenseRank,
        ops.MinRank,
        ops.NTile,
        ops.PercentRank,
        ops.CumeDist,
        ops.RowNumber,
    )

    if frame.start is None and frame.end is None:
        # no-op, default is full sample
        pass
    elif not isinstance(func, frame_clause_not_allowed):
        if frame.start is None:
            start = "UNBOUNDED PRECEDING"
        else:
            start = format_window_boundary(frame.start, **kw)

        if frame.end is None:
            end = "UNBOUNDED FOLLOWING"
        else:
            end = format_window_boundary(frame.end, **kw)

        frame = f"{frame.how.upper()} BETWEEN {start} AND {end}"
        components.append(frame)

    return f"OVER ({' '.join(components)})"


# TODO
_map_interval_to_microseconds = {
    "W": 604800000000,
    "D": 86400000000,
    "h": 3600000000,
    "m": 60000000,
    "s": 1000000,
    "ms": 1000,
    "us": 1,
    "ns": 0.001,
}


@translate_val.register(ops.ApproxMedian)
def _approx_median(op, **kw):
    expr = sg.func("approx_quantile", "0.5", translate_val(op.arg))
    return _apply_agg_filter(expr, where=op.where, **kw)


# TODO
@translate_val.register(ops.WindowFunction)
def _window(op: ops.WindowFunction, **kw: Any):
    if isinstance(op.func, ops.CumulativeOp):
        arg = cumulative_to_window(op.func, op.frame)
        return translate_val(arg, **kw)

    window_formatted = format_window_frame(op, op.frame, **kw)
    func = op.func.__window_op__
    func_formatted = translate_val(func, **kw)
    result = f"{func_formatted} {window_formatted}"

    if isinstance(func, ops.RankBase):
        return f"({result} - 1)"

    return result


def shift_like(op_class, name):
    @translate_val.register(op_class)
    def formatter(op, **kw):
        arg = op.arg
        offset = op.offset
        default = op.default

        arg_fmt = translate_val(arg, **kw)
        pieces = [arg_fmt]

        if default is not None:
            if offset is None:
                offset_fmt = "1"
            else:
                offset_fmt = translate_val(offset, **kw)

            default_fmt = translate_val(default, **kw)

            pieces.append(offset_fmt)
            pieces.append(default_fmt)
        elif offset is not None:
            offset_fmt = translate_val(offset, **kw)
            pieces.append(offset_fmt)

        return sg.func(name, *pieces)

    return formatter


shift_like(ops.Lag, "lag")
shift_like(ops.Lead, "lead")


@translate_val.register(ops.Argument)
def _argument(op, **_):
    return sg.exp.Identifier(this=op.name, quoted=False)


@translate_val.register(ops.JSONGetItem)
def _json_getitem(op, **kw):
    return sg.exp.JSONExtract(
        this=translate_val(op.arg, **kw), expression=translate_val(op.index, **kw)
    )


@translate_val.register(ops.RowID)
def _rowid(op, *, aliases, **_) -> str:
    table = op.table
    return sg.column(op.name, (aliases or {}).get(table, table.name))


@translate_val.register(ops.ScalarUDF)
def _scalar_udf(op, **kw) -> str:
    funcname = op.__class__.__name__
    return sg.func(funcname, *(translate_val(arg, **kw) for arg in op.args))
