from __future__ import annotations

import calendar
import contextlib
import functools
import math
from functools import partial
import operator
from operator import add, mul, sub
from typing import Any, Literal, Mapping

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import sqlglot as sg
from ibis.backends.base.sql.registry import helpers
from ibis.backends.duckdb.datatypes import serialize
from toolz import flip

# TODO: Ideally we can translate bottom up a la `relations.py`
# TODO: Find a way to remove all the dialect="duckdb" kwargs


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_val.register(dt.DataType)
def _datatype(t, **_):
    return serialize(t)


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
        return sg.alias(val, op.name, dialect="duckdb")
    return val


### Bitwise Business

_bitwise_mapping = {
    ops.BitwiseLeftShift: "<<",
    ops.BitwiseRightShift: ">>",
    ops.BitwiseAnd: "&",
    ops.BitwiseOr: "|",
}


@translate_val.register(ops.BitwiseLeftShift)
@translate_val.register(ops.BitwiseRightShift)
@translate_val.register(ops.BitwiseAnd)
@translate_val.register(ops.BitwiseOr)
def _bitwise_binary(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    _operator = _bitwise_mapping[type(op)]

    return f"{left} {_operator} {right}"


@translate_val.register(ops.BitwiseXor)
def _bitwise_xor(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)

    return f"xor({left}, {right})"


@translate_val.register(ops.BitwiseNot)
def _bitwise_not(op, **kw):
    value = translate_val(op.arg, **kw)

    return f"~{value}"


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
        return f"ln({arg}) / ln({base})"
    return f"ln({arg})"


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
            return f"INTERVAL {arg} {suffix}"
    elif isinstance(op.to, dt.Timestamp) and isinstance(op.arg.dtype, dt.Integer):
        return sg.func("to_timestamp", arg)
    elif isinstance(op.to, dt.Timestamp) and (timezone := op.to.timezone) is not None:
        return sg.func("timezone", timezone, arg)

    to = translate_val(op.to, **kw)
    return sg.cast(expression=arg, to=to)


@translate_val.register(ops.TryCast)
def _try_cast(op, **kw):
    return sg.func(
        "try_cast", translate_val(op.arg, **kw), serialize(op.to), dialect="duckdb"
    )


### Comparator Conundrums


@translate_val.register(ops.Between)
def _between(op, **kw):
    arg = translate_val(op.arg, **kw)
    lower_bound = translate_val(op.lower_bound, **kw)
    upper_bound = translate_val(op.upper_bound, **kw)
    return f"{arg} BETWEEN {lower_bound} AND {upper_bound}"


@translate_val.register(ops.Negate)
def _negate(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"-{_parenthesize(op.arg, arg)}"


@translate_val.register(ops.Not)
def _not(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"NOT {_parenthesize(op.arg, arg)}"


def _parenthesize(op, arg):
    # function calls don't need parens
    if isinstance(op, (ops.Binary, ops.Unary)):
        return f"({arg})"
    else:
        return arg


### Timey McTimeFace


@translate_val.register(ops.Date)
def _to_date(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"DATE {arg}"


@translate_val.register(ops.Time)
def _time(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"{arg}::TIME"


@translate_val.register(ops.Strftime)
def _strftime(op, **kw):
    if not isinstance(op.format_str, ops.Literal):
        raise com.UnsupportedOperationError(
            f"DuckDB format_str must be a literal `str`; got {type(op.format_str)}"
        )
    arg = translate_val(op.arg, **kw)
    format_str = translate_val(op.format_str, **kw)
    return sg.func("strftime", arg, format_str)


@translate_val.register(ops.TimeFromHMS)
def _time_from_hms(op, **kw):
    hours = translate_val(op.hours, **kw)
    minutes = translate_val(op.minutes, **kw)
    seconds = translate_val(op.seconds, **kw)
    return sg.func("make_time", hours, minutes, seconds)


@translate_val.register(ops.StringToTimestamp)
def _string_to_timestamp(op, **kw):
    arg = translate_val(op.arg, **kw)
    format_str = translate_val(op.format_str, **kw)
    return sg.func("strptime", arg, format_str)


@translate_val.register(ops.ExtractEpochSeconds)
def _extract_epoch_seconds(op, **kw):
    arg = translate_val(op.arg, **kw)
    # TODO: do we need the TIMESTAMP cast?
    return f"epoch({arg}::TIMESTAMP)"


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
    return f"extract({part}, {timestamp})"


# DuckDB extracts subminute microseconds and milliseconds
# so we have to finesse it a little bit
@translate_val.register(ops.ExtractMicrosecond)
def _extract_microsecond(op, **kw):
    arg = translate_val(op.arg, **kw)
    dtype = serialize(op.dtype)

    return f"extract('us', {arg}::TIMESTAMP) % 1000000"


@translate_val.register(ops.ExtractMillisecond)
def _extract_microsecond(op, **kw):
    arg = translate_val(op.arg, **kw)
    dtype = serialize(op.dtype)

    return f"extract('ms', {arg}::TIMESTAMP) % 1000"


@translate_val.register(ops.Date)
def _date(op, **kw):
    arg = translate_val(op.arg, **kw)

    return f"{arg}::DATE"


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


@translate_val.register(ops.DateFromYMD)
def _date_from_ymd(op, **kw):
    y = translate_val(op.year, **kw)
    m = translate_val(op.month, **kw)
    d = translate_val(op.day, **kw)
    return f"make_date({y}, {m}, {d})"


@translate_val.register(ops.DayOfWeekIndex)
def _day_of_week_index(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"(dayofweek({arg}) + 6) % 7"


@translate_val.register(ops.TimestampFromUNIX)
def _timestamp_from_unix(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (unit := op.unit.short) in {"ns"}:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    if op.unit.short == "ms":
        return f"to_timestamp({arg[:-3]}) + INTERVAL {arg[-3:]} millisecond"
    elif op.unit.short == "us":
        return f"to_timestamp({arg[:-6]}) + INTERVAL {arg[-6:]} microsecond"

    return f"to_timestamp({arg})"


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


### Interval Marginalia


_interval_mapping = {
    ops.IntervalAdd: operator.add,
    ops.IntervalSubtract: operator.sub,
    ops.IntervalMultiply: operator.mul,
}


@translate_val.register(ops.IntervalAdd)
@translate_val.register(ops.IntervalSubtract)
@translate_val.register(ops.IntervalMultiply)
def _interval_binary(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    _operator = _interval_mapping[type(op)]

    return operator(left, right)


def _interval_format(op):
    dtype = op.dtype
    if dtype.unit.short == "ns":
        raise com.UnsupportedOperationError(
            "Duckdb doesn't support nanosecond interval resolutions"
        )

    return f"INTERVAL {op.value} {dtype.resolution.upper()}"


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
    # TODO: make less gross
    # to_days, to_years, etc...
    return sg.func(f"to_{op.dtype.resolution}s", arg)


### String Instruments


@translate_val.register(ops.Substring)
def _substring(op, **kw):
    # Duckdb is 1-indexed
    arg = translate_val(op.arg, **kw)
    start = translate_val(op.start, **kw)
    arg_length = f"length({arg})"
    if op.length is not None:
        length = translate_val(op.length, **kw)
        suffix = f", {length}"
    else:
        suffix = ""

    if_pos = f"substring({arg}, {start} + 1{suffix})"
    if_neg = f"substring({arg}, {arg_length} + {start} + 1{suffix})"
    return f"if({start} >= 0, {if_pos}, {if_neg})"


@translate_val.register(ops.StringFind)
def _string_find(op, **kw):
    if op.end is not None:
        raise com.UnsupportedOperationError("String find doesn't support end argument")

    arg = translate_val(op.arg, **kw)
    substr = translate_val(op.substr, **kw)

    return f"instr({arg}, {substr}) - 1"


@translate_val.register(ops.RegexSearch)
def _regex_search(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return f"regexp_matches({arg}, {pattern}, 's')"


@translate_val.register(ops.RegexReplace)
def _regex_replace(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    replacement = translate_val(op.replacement, **kw)
    return sg.func("regexp_replace", arg, pattern, replacement, "g", dialect="duckdb")


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    group = translate_val(op.index, **kw)
    return f"regexp_extract({arg}, {pattern}, {group})"


@translate_val.register(ops.Levenshtein)
def _levenshtein(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return f"levenshtein({left}, {right})"


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
    # ops.ApproxMedian: "median",  # TODO
    # ops.Median: "quantileExactExclusive",  # TODO
    ops.ApproxCountDistinct: "list_unique",
    ops.Mean: "avg",
    ops.Sum: "sum",
    ops.Max: "max",
    ops.Min: "min",
    ops.Any: "any_value",
    ops.All: "min",
    ops.ArgMin: "arg_min",
    ops.Mode: "mode",
    ops.ArgMax: "arg_max",
    # ops.ArrayCollect: "groupArray",  # TODO
    ops.Count: "count",
    ops.CountDistinct: "list_unique",
    ops.First: "first",
    ops.Last: "last",
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
    # Temporal operations
    ops.TimestampNow: "current_timestamp",
    # Other operations
    ops.Where: "if",
    ops.ArrayLength: "length",
    ops.ArrayConcat: "arrayConcat",  # TODO
    ops.Unnest: "arrayJoin",  # TODO
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.NullIf: "nullIf",
    ops.MapContains: "mapContains",  # TODO
    ops.MapLength: "length",
    ops.MapKeys: "mapKeys",  # TODO
    ops.MapValues: "mapValues",  # TODO
    ops.MapMerge: "mapUpdate",  # TODO
    ops.ArrayDistinct: "arrayDistinct",  # TODO
    ops.ArraySort: "arraySort",  # TODO
    ops.ArrayContains: "has",
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
            return sg.func(
                _name, *map(partial(translate_val, **kw), op.args), dialect="duckdb"
            )


del _fmt, _name, _op


### NULL PLAYER CHARACTER
# ops.IsNull: "isNull",  # TODO
# ops.NotNull: "isNotNull",  # TODO
# ops.IfNull: "ifNull",  # TODO
@translate_val.register(ops.IsNull)
def _is_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return arg.is_(sg.expressions.null())


@translate_val.register(ops.NotNull)
def _is_not_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return arg.is_(sg.not_(sg.expressions.null()))


@translate_val.register(ops.IfNull)
def _if_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    ifnull = translate_val(op.ifnull_expr, **kw)
    return sg.func("ifnull", arg, ifnull, dialect="duckdb")


### Definitely Not Tensors


@translate_val.register(ops.ArrayIndex)
def _array_index_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    index = translate_val(op.index, **kw)
    correct_idx = f"if({index} >= 0, {index} + 1, {index})"
    return f"array_extract({arg}, {correct_idx})"


@translate_val.register(ops.InValues)
def _in_values(op, **kw):
    if not op.options:
        return False
    value = translate_val(op.value, **kw)
    options = [translate_val(x, **kw) for x in op.options]
    return sg.func("list_contains", options, value, dialect="duckdb")


@translate_val.register(ops.InColumn)
def _in_column(op, **kw):
    value = translate_val(op.value, **kw)
    options = translate_val(ops.TableArrayView(op.options.to_expr().as_table()), **kw)
    # TODO: fix?
    # if not isinstance(options, sa.sql.Selectable):
    #     options = sg.select(options)
    return value.isin(options)


### LITERALLY


# TODO: need to go through this carefully
@translate_val.register(ops.Literal)
def _literal(op, **kw):
    value = op.value
    dtype = op.dtype
    if value is None and dtype.nullable:
        if dtype.is_null():
            return "Null"
        return f"CAST(Null AS {serialize(dtype)})"
    if dtype.is_boolean():
        return str(int(bool(value)))
    elif dtype.is_inet():
        com.UnsupportedOperationError("DuckDB doesn't support an explicit inet dtype")
    elif dtype.is_string():
        return value
    elif dtype.is_decimal():
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

        # TODO: handle if `value` is "Infinity"

        return f"{value!s}::decimal({precision}, {scale})"
    elif dtype.is_numeric():
        if math.isinf(value):
            return f"'{repr(value)}inity'::FLOAT"
        elif math.isnan(value):
            return "'NaN'::FLOAT"
        return value
    elif dtype.is_interval():
        return _interval_format(op)
    elif dtype.is_timestamp():
        year = op.value.year
        month = op.value.month
        day = op.value.day
        hour = op.value.hour
        minute = op.value.minute
        second = op.value.second
        if op.value.microsecond:
            microsecond = op.value.microsecond / 1e6
            second += microsecond
        if (timezone := dtype.timezone) is not None:
            return f"make_timestamptz({year}, {month}, {day}, {hour}, {minute}, {second}, '{timezone}')"
        else:
            return f"make_timestamp({year}, {month}, {day}, {hour}, {minute}, {second})"
    elif dtype.is_date():
        return f"make_date({op.value.year}, {op.value.month}, {op.value.day})"
    elif dtype.is_array():
        value_type = dtype.value_type
        values = ", ".join(
            _literal(ops.Literal(v, dtype=value_type), **kw) for v in value
        )
        return f"[{values}]"
    elif dtype.is_map():
        value_type = dtype.value_type
        values = ", ".join(
            f"{k!r}, {_literal(ops.Literal(v, dtype=value_type), **kw)}"
            for k, v in value.items()
        )
        return f"map({values})"
    elif dtype.is_struct():
        fields = ", ".join(
            _literal(ops.Literal(v, dtype=subdtype), **kw)
            for subdtype, v in zip(dtype.types, value.values())
        )
        return f"tuple({fields})"
    else:
        raise NotImplementedError(f"Unsupported type: {dtype!r}")


### BELOW HERE BE DRAGONS


# TODO
@translate_val.register(ops.ArrayRepeat)
def _array_repeat_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    from_ = f"(SELECT {arg} AS arr FROM system.numbers LIMIT {times})"
    query = sg.parse_one(
        f"SELECT arrayFlatten(groupArray(arr)) FROM {from_}", read="duckdb"
    )
    return query.subquery()


# TODO
@translate_val.register(ops.ArraySlice)
def _array_slice_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    start = translate_val(op.start, **kw)
    start = _parenthesize(op.start, start)
    start_correct = f"if({start} < 0, {start}, {start} + 1)"

    if (stop := op.stop) is not None:
        stop = translate_val(stop, **kw)
        stop = _parenthesize(op.stop, stop)

        neg_start = f"(length({arg}) + {start})"
        diff_fmt = f"greatest(-0, {stop} - {{}})".format

        length = (
            f"if({stop} < 0, {stop}, "
            f"if({start} < 0, {diff_fmt(neg_start)}, {diff_fmt(start)}))"
        )

        return f"arraySlice({arg}, {start_correct}, {length})"

    return f"arraySlice({arg}, {start_correct})"


@translate_val.register(ops.CountStar)
def _count_star(op, **kw):
    sql = sg.expressions.Count(this=sg.expressions.Star())
    if (predicate := op.where) is not None:
        return sg.select(sql).where(predicate)
    return sql


@translate_val.register(ops.NotAny)
def _not_any(op, **kw):
    return translate_val(ops.All(ops.Not(op.arg), where=op.where), **kw)


@translate_val.register(ops.NotAll)
def _not_all(op, **kw):
    return translate_val(ops.Any(ops.Not(op.arg), where=op.where), **kw)


# TODO
def _quantile_like(func_name: str, op: ops.Node, quantile: str, **kw):
    args = [_sql(translate_val(op.arg, **kw))]

    if (where := op.where) is not None:
        args.append(_sql(translate_val(where, **kw)))
        func_name += "If"

    return f"{func_name}({quantile})({', '.join(args)})"


@translate_val.register(ops.Quantile)
def _quantile(op, **kw):
    quantile = _sql(translate_val(op.quantile, **kw))
    return _quantile_like("quantile", op, quantile, **kw)


@translate_val.register(ops.MultiQuantile)
def _multi_quantile(op, **kw):
    if not isinstance(op.quantile, ops.Literal):
        raise TypeError("Duckdb quantile only accepts a list of Python floats")

    quantile = ", ".join(map(str, op.quantile.value))
    return _quantile_like("quantiles", op, quantile, **kw)


def _agg_variance_like(func):
    variants = {"sample": f"{func}_samp", "pop": f"{func}_pop"}

    def formatter(op, **kw):
        return _aggregate(op, variants[op.how], where=op.where, **kw)

    return formatter


@translate_val.register(ops.Correlation)
def _corr(op, **kw):
    if op.how == "pop":
        raise ValueError("Duckdb only implements `sample` correlation coefficient")
    return _aggregate(op, "corr", where=op.where, **kw)


def _aggregate(op, func, *, where=None, **kw):
    args = [
        translate_val(arg, **kw)
        for argname, arg in zip(op.argnames, op.args)
        if argname not in ("where", "how")
    ]
    if where is not None:
        predicate = translate_val(where, **kw)
        return sg.func(func, *args).where(predicate)

    res = sg.func(func, *args)
    return res


@translate_val.register(ops.Arbitrary)
def _arbitrary(op, **kw):
    functions = {
        "first": "first",
        "last": "last",
    }
    return _aggregate(op, functions[op.how], where=op.where, **kw)


@translate_val.register(ops.FindInSet)
def _index_of(op, **kw):
    values = map(partial(translate_val, **kw), op.values)
    values = ", ".join(map(_sql, values))
    needle = translate_val(op.needle, **kw)
    return f"list_indexof([{values}], {needle}) - 1"


@translate_val.register(ops.Round)
def _round(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (digits := op.digits) is not None:
        return f"round({arg}, {translate_val(digits, **kw)})"
    return f"round({arg})"


@translate_val.register(tuple)
def _node_list(op, punct="()", **kw):
    values = ", ".join(map(_sql, map(partial(translate_val, **kw), op)))
    return f"{punct[0]}{values}{punct[1]}"


def _sql(obj, dialect="duckdb"):
    try:
        return obj.sql(dialect=dialect)
    except AttributeError:
        return obj


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, **kw):
    case = sg.expressions.Case()

    if (base := getattr(op, "base", None)) is not None:
        case = sg.expressions.Case(this=translate_val(base, **kw))

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

    foreign_table = translate_rel(op.foreign_table, **kw)
    predicates = translate_val(op.predicates, **kw)
    subq = (
        sg.select(1)
        .from_(foreign_table, dialect="duckdb")
        .where(sg.condition(predicates), dialect="duckdb")
    )
    prefix = "NOT " * isinstance(op, ops.NotExistsSubquery)
    return f"{prefix}EXISTS ({subq})"


@translate_val.register(ops.StringSplit)
def _string_split(op, **kw):
    arg = translate_val(op.arg, **kw)
    delimiter = translate_val(op.delimiter, **kw)
    return f"string_split({arg}, {delimiter})"


@translate_val.register(ops.StringJoin)
def _string_join(op, **kw):
    arg = map(partial(translate_val, **kw), op.arg)
    sep = translate_val(op.sep, **kw)
    elements = ", ".join(map(_sql, arg))
    return f"list_aggregate([{elements}], 'string_agg', {sep})"


@translate_val.register(ops.StringConcat)
def _string_concat(op, **kw):
    arg = map(partial(translate_val, **kw), op.arg)
    return " || ".join(map(_sql, arg))


@translate_val.register(ops.StringSQLLike)
def _string_like(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return f"{arg} LIKE {pattern}"


@translate_val.register(ops.StringSQLILike)
def _string_ilike(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return f"lower({arg}) LIKE lower({pattern})"


# TODO
@translate_val.register(ops.Capitalize)
def _string_capitalize(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"CONCAT(UPPER(SUBSTR({arg}, 1, 1)), LOWER(SUBSTR({arg}, 2)))"


# TODO
@translate_val.register(ops.GroupConcat)
def _group_concat(op, **kw):
    arg = translate_val(op.arg, **kw)
    sep = translate_val(op.sep, **kw)

    args = [arg]
    func = "groupArray"

    if (where := op.where) is not None:
        func += "If"
        args.append(translate_val(where, **kw))

    joined_args = ", ".join(map(_sql, args))
    call = f"{func}({joined_args})"
    expr = f"list_concat({call}, {sep})"
    return f"CASE WHEN empty({call}) THEN NULL ELSE {expr} END"


# TODO
def _bit_agg(func):
    def _translate(op, **kw):
        arg = translate_val(op.arg, **kw)
        if not isinstance((type := op.arg.dtype), dt.UnsignedInteger):
            nbits = type.nbytes * 8
            arg = f"reinterpretAsUInt{nbits}({arg})"

        if (where := op.where) is not None:
            return f"{func}If({arg}, {translate_val(where, **kw)})"
        else:
            return f"{func}({arg})"

    return _translate


@translate_val.register(ops.ArrayColumn)
def _array_column(op, **kw):
    cols = map(partial(translate_val, **kw), op.cols)
    args = ", ".join(map(_sql, cols))
    return f"[{args}]"


# TODO
@translate_val.register(ops.StructColumn)
def _struct_column(op, **kw):
    values = translate_val(op.values, **kw)
    struct_type = serialize(op.dtype.copy(nullable=False))
    return f"CAST({values} AS {struct_type})"


@translate_val.register(ops.Clip)
def _clip(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (upper := op.upper) is not None:
        arg = f"least({translate_val(upper, **kw)}, {arg})"

    if (lower := op.lower) is not None:
        arg = f"greatest({translate_val(lower, **kw)}, {arg})"

    return arg


@translate_val.register(ops.StructField)
def _struct_field(op, render_aliases: bool = False, **kw):
    arg = op.arg
    arg_dtype = arg.dtype
    arg = translate_val(op.arg, render_aliases=render_aliases, **kw)
    idx = arg_dtype.names.index(op.field)
    typ = arg_dtype.types[idx]
    return f"CAST({arg}.{idx + 1} AS {serialize(typ)})"


# TODO
@translate_val.register(ops.NthValue)
def _nth_value(op, **kw):
    arg = translate_val(op.arg, **kw)
    nth = translate_val(op.nth, **kw)
    return f"nth_value({arg}, ({nth}) + 1)"


@translate_val.register(ops.Repeat)
def _repeat(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    return f"repeat({arg}, {times})"


# TODO
@translate_val.register(ops.NullIfZero)
def _null_if_zero(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf({arg}, 0)"


# TODO
@translate_val.register(ops.ZeroIfNull)
def _zero_if_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"ifNull({arg}, 0)"


@translate_val.register(ops.FloorDivide)
def _floor_divide(op, **kw):
    new_op = ops.Floor(ops.Divide(op.left, op.right))
    return translate_val(new_op, **kw)


@translate_val.register(ops.ScalarParameter)
def _scalar_param(op, params: Mapping[ops.Node, Any], **kw):
    raw_value = params[op]
    dtype = op.dtype
    if isinstance(dtype, dt.Struct):
        literal = ibis.struct(raw_value, type=dtype)
    elif isinstance(dtype, dt.Map):
        literal = ibis.map(raw_value, type=dtype)
    else:
        literal = ibis.literal(raw_value, type=dtype)
    return translate_val(literal.op(), **kw)


# TODO
def contains(op_string: Literal["IN", "NOT IN"]) -> str:
    def tr(op, *, cache, **kw):
        from ibis.backends.duckdb.compiler import translate

        value = op.value
        options = op.options
        if isinstance(options, tuple) and not options:
            return {"NOT IN": "TRUE", "IN": "FALSE"}[op_string]

        left_arg = translate_val(value, **kw)
        if helpers.needs_parens(value):
            left_arg = helpers.parenthesize(left_arg)

        # special case non-foreign isin/notin expressions
        if (
            not isinstance(options, tuple)
            and options.output_shape is rlz.Shape.COLUMNAR
        ):
            # this will fail to execute if there's a correlation, but it's too
            # annoying to detect so we let it through to enable the
            # uncorrelated use case (pandas-style `.isin`)
            subquery = translate(options.to_expr().as_table().op(), {})
            right_arg = f"({_sql(subquery)})"
        else:
            right_arg = _sql(translate_val(options, cache=cache, **kw))

        # we explicitly do NOT parenthesize the right side because it doesn't
        # make sense to do so for Sequence operations
        return f"{left_arg} {op_string} {right_arg}"

    return tr


# TODO
# translate_val.register(ops.Contains)(contains("IN"))
# translate_val.register(ops.NotContains)(contains("NOT IN"))


# TODO
@translate_val.register(ops.DayOfWeekName)
def day_of_week_name(op, **kw):
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


# TODO
@translate_val.register(ops.Map)
def _map(op, **kw):
    keys = translate_val(op.keys, **kw)
    values = translate_val(op.values, **kw)
    typ = serialize(op.dtype)
    return f"CAST(({keys}, {values}) AS {typ})"


# TODO
@translate_val.register(ops.MapGet)
def _map_get(op, **kw):
    arg = translate_val(op.arg, **kw)
    key = translate_val(op.key, **kw)
    default = translate_val(op.default, **kw)
    return f"if(mapContains({arg}, {key}), {arg}[{key}], {default})"


def _binary_infix(symbol: str):
    def formatter(op, **kw):
        left = translate_val(op_left := op.left, **kw)
        right = translate_val(op_right := op.right, **kw)

        return symbol(left, right)

    return formatter


import operator

_binary_infix_ops = {
    # Binary operations
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.Modulus: operator.mod,
    # Comparisons
    ops.GreaterEqual: operator.ge,
    ops.Greater: operator.gt,
    ops.LessEqual: operator.le,
    ops.Less: operator.lt,
    # Boolean comparisons
    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.DateAdd: operator.add,
    ops.DateSub: operator.sub,
    ops.DateDiff: operator.sub,
    ops.TimestampAdd: operator.add,
    ops.TimestampSub: operator.sub,
    ops.TimestampDiff: operator.sub,
}


for _op, _sym in _binary_infix_ops.items():
    translate_val.register(_op)(_binary_infix(_sym))

del _op, _sym


@translate_val.register(ops.Equals)
def _equals(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    return left.eq(right)


@translate_val.register(ops.NotEquals)
def _equals(op, **kw):
    left = translate_val(op.left, **kw)
    right = translate_val(op.right, **kw)
    breakpoint()
    return left.eq(right)


# TODO
translate_val.register(ops.BitAnd)(_bit_agg("groupBitAnd"))
translate_val.register(ops.BitOr)(_bit_agg("groupBitOr"))
translate_val.register(ops.BitXor)(_bit_agg("groupBitXor"))

translate_val.register(ops.StandardDev)(_agg_variance_like("stddev"))
translate_val.register(ops.Variance)(_agg_variance_like("var"))
translate_val.register(ops.Covariance)(_agg_variance_like("covar"))


@translate_val.register
def _sort_key(op: ops.SortKey, **kw):
    arg = translate_val(op.expr, **kw)
    direction = "ASC" if op.ascending else "DESC"
    return f"{_sql(arg)} {direction}"


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


# TODO
UNSUPPORTED_REDUCTIONS = (
    ops.ApproxMedian,
    ops.GroupConcat,
    ops.ApproxCountDistinct,
)


# TODO
@translate_val.register(ops.WindowFunction)
def _window(op: ops.WindowFunction, **kw: Any):
    if isinstance(op.func, UNSUPPORTED_REDUCTIONS):
        raise com.UnsupportedOperationError(
            f"{type(op.func)} is not supported in window functions"
        )

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


# TODO
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

        return f"{name}({', '.join(map(_sql, pieces))})"

    return formatter


# TODO
shift_like(ops.Lag, "lagInFrame")
shift_like(ops.Lead, "leadInFrame")


# TODO
@translate_val.register(ops.RowNumber)
def _row_number(_, **kw):
    return "row_number()"


# TODO
@translate_val.register(ops.DenseRank)
def _dense_rank(_, **kw):
    return "dense_rank()"


# TODO
@translate_val.register(ops.MinRank)
def _rank(_, **kw):
    return "rank()"


@translate_val.register(ops.ArrayStringJoin)
def _array_string_join(op, **kw):
    arg = translate_val(op.arg, **kw)
    sep = translate_val(op.sep, **kw)
    return f"list_aggregate({arg}, 'string_agg', {sep})"


@translate_val.register(ops.Argument)
def _argument(op, **_):
    return op.name


@translate_val.register(ops.ArrayMap)
def _array_map(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    return sg.func("list_transform", arg, f"{op.parameter}) -> {result}")


@translate_val.register(ops.ArrayFilter)
def _array_filter(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    return sg.func("list_filter", arg, f"{op.parameter} -> {result}")


@translate_val.register(ops.ArrayPosition)
def _array_position(op, **kw):
    arg = translate_val(op.arg, **kw)
    el = translate_val(op.other, **kw)
    return f"list_indexof({arg}, {el}) - 1"


@translate_val.register(ops.ArrayRemove)
def _array_remove(op, **kw):
    return translate_val(ops.ArrayFilter(op.arg, flip(ops.NotEquals, op.other)), **kw)


@translate_val.register(ops.ArrayUnion)
def _array_union(op, **kw):
    return translate_val(ops.ArrayDistinct(ops.ArrayConcat(op.left, op.right)), **kw)


# TODO
@translate_val.register(ops.ArrayZip)
def _array_zip(op: ops.ArrayZip, **kw: Any) -> str:
    arglist = []
    for arg in op.arg:
        sql_arg = translate_val(arg, **kw)
        with contextlib.suppress(AttributeError):
            sql_arg = sql_arg.sql(dialect="duckdb")
        arglist.append(sql_arg)
    return f"arrayZip({', '.join(arglist)})"
