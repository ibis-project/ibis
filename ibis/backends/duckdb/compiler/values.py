from __future__ import annotations

import calendar
import functools
import math
import operator
import string
from functools import partial
from typing import Any

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import NULL, STAR, AggGen, FuncGen, make_cast
from ibis.backends.base.sqlglot.datatypes import DuckDBType


def _aggregate(funcname, *args, where):
    expr = f[funcname](*args)
    if where is not None:
        return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
    return expr


f = FuncGen()
if_ = f["if"]
cast = make_cast(DuckDBType)
agg = AggGen(aggfunc=_aggregate)


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_val.register(ops.Field)
def _field(op, *, rel, name, **_):
    return sg.column(name, table=rel.alias_or_name)


@translate_val.register(ops.ScalarSubquery)
def _subquery(op, *, rel, **_):
    return rel.this.subquery()


@translate_val.register(ops.Alias)
def _alias(op, *, arg, name, **_):
    return arg.as_(name)


### Literals


@translate_val.register(ops.Literal)
def _literal(op, *, value, dtype, **kw):
    if value is None:
        if dtype.nullable:
            return NULL if dtype.is_null() else cast(NULL, dtype)
        raise NotImplementedError(f"Unsupported NULL for non-nullable type: {dtype!r}")
    elif dtype.is_interval():
        if dtype.unit.short == "ns":
            raise com.UnsupportedOperationError(
                "Duckdb doesn't support nanosecond interval resolutions"
            )

        return sg.exp.Interval(
            this=sg.exp.convert(str(value)), unit=dtype.resolution.upper()
        )
    elif dtype.is_boolean():
        return sg.exp.Boolean(this=value)
    elif dtype.is_string() or dtype.is_inet() or dtype.is_macaddr():
        return sg.exp.convert(str(value))
    elif dtype.is_numeric():
        # cast non finite values to float because that's the behavior of
        # duckdb when a mixed decimal/float operation is performed
        #
        # float will be upcast to double if necessary by duckdb
        if not math.isfinite(value):
            return cast(str(value), to=dt.float32 if dtype.is_decimal() else dtype)
        return cast(value, dtype)
    elif dtype.is_time():
        return f.make_time(
            value.hour, value.minute, value.second + value.microsecond / 1e6
        )
    elif dtype.is_timestamp():
        args = [
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second + value.microsecond / 1e6,
        ]

        func = "make_timestamp"
        if (tz := dtype.timezone) is not None:
            func += "tz"
            args.append(tz)

        return f[func](*args)
    elif dtype.is_date():
        return sg.exp.DateFromParts(
            year=sg.exp.convert(value.year),
            month=sg.exp.convert(value.month),
            day=sg.exp.convert(value.day),
        )
    elif dtype.is_array():
        value_type = dtype.value_type
        make_value = partial(_literal, dtype=value_type, **kw)
        return f.array(
            *(make_value(ops.Literal(v, dtype=value_type), value=v) for v in value)
        )
    elif dtype.is_map():
        key_type = dtype.key_type
        make_key = partial(_literal, dtype=key_type, **kw)
        keys = f.array(
            *(make_key(ops.Literal(k, dtype=key_type), value=k) for k in value.keys())
        )

        value_type = dtype.value_type
        make_value = partial(_literal, dtype=value_type, **kw)
        values = f.array(
            *(
                make_value(ops.Literal(v, dtype=value_type), value=v)
                for v in value.values()
            )
        )

        return sg.exp.Map(keys=keys, values=values)
    elif dtype.is_struct():
        make_field = partial(_literal, **kw)
        items = [
            sg.exp.Slice(
                this=sg.exp.convert(k),
                expression=make_field(
                    ops.Literal(v, dtype=field_dtype), value=v, dtype=field_dtype
                ),
            )
            for field_dtype, (k, v) in zip(dtype.types, value.items())
        ]
        return sg.exp.Struct.from_arg_list(items)
    elif dtype.is_uuid():
        return cast(str(value), dtype)
    elif dtype.is_binary():
        return cast("".join(map("\\x{:02x}".format, value)), dtype)
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
    ops.All: "bool_and",
    ops.Any: "bool_or",
    ops.ArrayCollect: "list",
    ops.GroupConcat: "string_agg",
    ops.BitOr: "bit_or",
    ops.BitAnd: "bit_and",
    ops.BitXor: "bit_xor",
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
    ops.StringAscii: "ascii",
    ops.StrRight: "right",
    # Other operations
    ops.IfElse: "if",
    ops.ArrayLength: "length",
    ops.Unnest: "unnest",
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.NullIf: "nullif",
    ops.MapLength: "cardinality",
    ops.MapKeys: "map_keys",
    ops.MapValues: "map_values",
    ops.ArraySort: "list_sort",
    ops.ArrayContains: "list_contains",
    ops.FirstValue: "first_value",
    ops.LastValue: "last_value",
    ops.NTile: "ntile",
    ops.Hash: "hash",
    ops.TimeFromHMS: "make_time",
    ops.StringToTimestamp: "strptime",
    ops.Levenshtein: "levenshtein",
    ops.Repeat: "repeat",
    ops.Map: "map",
    ops.MapMerge: "map_concat",
    ops.JSONGetItem: "json_extract",
    ops.TypeOf: "typeof",
    ops.IntegerRange: "range",
    ops.ArrayFlatten: "flatten",
    ops.ArrayPosition: "list_indexof",
}


for _op, _name in _simple_ops.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @translate_val.register(_op)
        def _fmt(op, _name: str = _name, *, where, **kw):
            return agg[_name](*kw.values(), where=where)

    else:

        @translate_val.register(_op)
        def _fmt(op, _name: str = _name, **kw):
            return f[_name](*kw.values())


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
def _bitwise_binary(op, *, left, right, **_):
    sg_expr = _bitwise_mapping[type(op)]
    return sg_expr(this=left, expression=right)


@translate_val.register(ops.BitwiseNot)
def _bitwise_not(op, *, arg, **_):
    return sg.exp.BitwiseNot(this=arg)


### Mathematical Calisthenics


@translate_val.register(ops.E)
def _euler(op, **_):
    return f.exp(1)


@translate_val.register(ops.Log)
def _generic_log(op, *, arg, base, **_):
    if base is None:
        return f.ln(arg)
    elif str(base) in ("2", "10"):
        return f[f"log{base}"](arg)
    else:
        return f.ln(arg) / f.ln(base)


@translate_val.register(ops.Clip)
def _clip(op, *, arg, lower, upper, **_):
    if upper is not None:
        arg = if_(arg.is_(NULL), arg, f.least(upper, arg))

    if lower is not None:
        arg = if_(arg.is_(NULL), arg, f.greatest(lower, arg))

    return arg


@translate_val.register(ops.FloorDivide)
def _floor_divide(op, *, left, right, **_):
    return cast(f.fdiv(left, right), op.dtype)


@translate_val.register(ops.Round)
def _round(op, *, arg, digits, **_):
    if digits is not None:
        return sg.exp.Round(this=arg, decimals=digits)
    return sg.exp.Round(this=arg)


### Dtype Dysmorphia


_interval_suffixes = {
    "ms": "milliseconds",
    "us": "microseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "D": "days",
    "M": "months",
    "Y": "years",
}


@translate_val.register(ops.Cast)
def _cast(op, *, arg, to, **_):
    if to.is_interval():
        return f[f"to_{_interval_suffixes[to.unit.short]}"](
            sg.cast(arg, to=DuckDBType.from_ibis(dt.int32))
        )
    elif to.is_timestamp() and op.arg.dtype.is_integer():
        return f.to_timestamp(arg)

    return cast(arg, to)


@translate_val.register(ops.TryCast)
def _try_cast(op, *, arg, to, **_):
    return sg.exp.TryCast(this=arg, to=DuckDBType.from_ibis(to))


### Comparator Conundrums


@translate_val.register(ops.Between)
def _between(op, *, arg, lower_bound, upper_bound, **_):
    return sg.exp.Between(this=arg, low=lower_bound, high=upper_bound)


@translate_val.register(ops.Negate)
def _negate(op, *, arg, **_):
    return sg.exp.Neg(this=arg)


@translate_val.register(ops.Not)
def _not(op, *, arg, **_):
    return sg.exp.Not(this=arg)


### Timey McTimeFace


@translate_val.register(ops.Date)
def _to_date(op, *, arg, **_):
    return sg.exp.Date(this=arg)


@translate_val.register(ops.DateFromYMD)
def _date_from_ymd(op, *, year, month, day, **_):
    return sg.exp.DateFromParts(year=year, month=month, day=day)


@translate_val.register(ops.Time)
def _time(op, *, arg, **_):
    return cast(arg, to=dt.time)


@translate_val.register(ops.TimestampNow)
def _timestamp_now(op, **_):
    """DuckDB current timestamp defaults to timestamp + tz."""
    return cast(f.current_timestamp(), dt.timestamp)


_POWERS_OF_TEN = {
    "s": 0,
    "ms": 3,
    "us": 6,
    "ns": 9,
}


@translate_val.register(ops.TimestampFromUNIX)
def _timestamp_from_unix(op, *, arg, unit, **_):
    unit = unit.short
    if unit == "ms":
        return f.epoch_ms(arg)
    elif unit == "s":
        return sg.exp.UnixToTime(this=arg)
    else:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")


@translate_val.register(ops.TimestampFromYMDHMS)
def _timestamp_from_ymdhms(op, *, year, month, day, hours, minutes, seconds, **_):
    args = [year, month, day, hours, minutes, seconds]

    func = "make_timestamp"
    if (timezone := op.dtype.timezone) is not None:
        func += "tz"
        args.append(timezone)

    return f[func](*args)


@translate_val.register(ops.Strftime)
def _strftime(op, *, arg, format_str, **_):
    if not isinstance(op.format_str, ops.Literal):
        raise com.UnsupportedOperationError(
            f"DuckDB format_str must be a literal `str`; got {type(op.format_str)}"
        )
    return f.strftime(arg, format_str)


@translate_val.register(ops.ExtractEpochSeconds)
def _extract_epoch_seconds(op, *, arg, **_):
    return f.epoch(cast(arg, dt.timestamp))


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
def _extract_time(op, *, arg, **_):
    part = _extract_mapping[type(op)]
    return f.extract(part, arg)


# DuckDB extracts subminute microseconds and milliseconds
# so we have to finesse it a little bit
@translate_val.register(ops.ExtractMicrosecond)
def _extract_microsecond(op, *, arg, **_):
    return f.mod(f.extract("us", arg), 1_000_000)


@translate_val.register(ops.ExtractMillisecond)
def _extract_microsecond(op, *, arg, **_):
    return f.mod(f.extract("ms", arg), 1_000)


@translate_val.register(ops.DateTruncate)
@translate_val.register(ops.TimestampTruncate)
@translate_val.register(ops.TimeTruncate)
def _truncate(op, *, arg, unit, **_):
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

    unit = unit.short
    try:
        duckunit = unit_mapping[unit]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

    return f.date_trunc(duckunit, arg)


@translate_val.register(ops.DayOfWeekIndex)
def _day_of_week_index(op, *, arg, **_):
    return (f.dayofweek(arg) + 6) % 7


@translate_val.register(ops.DayOfWeekName)
def day_of_week_name(op, *, arg, **_):
    # day of week number is 0-indexed
    # Sunday == 0
    # Saturday == 6
    weekdays = range(7)
    return sg.exp.Case(
        this=(f.dayofweek(arg) + 6) % 7,
        ifs=[if_(day, calendar.day_name[day]) for day in weekdays],
    )


### Interval Marginalia


_interval_mapping = {
    ops.IntervalAdd: operator.add,
    ops.IntervalSubtract: operator.sub,
    ops.IntervalMultiply: operator.mul,
}


@translate_val.register(ops.IntervalAdd)
@translate_val.register(ops.IntervalSubtract)
@translate_val.register(ops.IntervalMultiply)
def _interval_binary(op, *, left, right, **_):
    func = _interval_mapping[type(op)]
    return func(left, right)


@translate_val.register(ops.IntervalFromInteger)
def _interval_from_integer(op, *, arg, **_):
    dtype = op.dtype
    if dtype.unit.short == "ns":
        raise com.UnsupportedOperationError(
            "Duckdb doesn't support nanosecond interval resolutions"
        )

    if op.dtype.resolution == "week":
        return f.to_days(arg * 7)
    return f[f"to_{op.dtype.resolution}s"](arg)


### String Instruments


@translate_val.register(ops.Strip)
def _strip(op, *, arg, **_):
    return f.trim(arg, string.whitespace)


@translate_val.register(ops.RStrip)
def _rstrip(op, *, arg, **_):
    return f.rtrim(arg, string.whitespace)


@translate_val.register(ops.LStrip)
def _lstrip(op, *, arg, **_):
    return f.ltrim(arg, string.whitespace)


@translate_val.register(ops.Substring)
def _substring(op, *, arg, start, length, **_):
    if_pos = sg.exp.Substring(this=arg, start=start + 1, length=length)
    if_neg = sg.exp.Substring(this=arg, start=start, length=length)

    return if_(start >= 0, if_pos, if_neg)


@translate_val.register(ops.StringFind)
def _string_find(op, *, arg, substr, start, end, **_):
    if end is not None:
        raise com.UnsupportedOperationError(
            "String find doesn't support `end` argument"
        )

    if start is not None:
        arg = f.substr(arg, start + 1)
        pos = f.strpos(arg, substr)
        return if_(pos > 0, pos - 1 + start, -1)

    return f.strpos(arg, substr)


@translate_val.register(ops.RegexSearch)
def _regex_search(op, *, arg, pattern, **_):
    return f.regexp_matches(arg, pattern, "s")


@translate_val.register(ops.RegexReplace)
def _regex_replace(op, *, arg, pattern, replacement, **_):
    return f.regexp_replace(arg, pattern, replacement, "g")


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, *, arg, pattern, index, **_):
    return f.regexp_extract(arg, pattern, index, dialect="duckdb")


@translate_val.register(ops.StringSplit)
def _string_split(op, *, arg, delimiter, **_):
    return sg.exp.Split(this=arg, expression=delimiter)


@translate_val.register(ops.StringJoin)
def _string_join(op, *, arg, sep, **_):
    return f.list_aggr(f.array(*arg), "string_agg", sep)


@translate_val.register(ops.StringConcat)
def _string_concat(op, *, arg, **_):
    return sg.exp.Concat.from_arg_list(list(arg))


@translate_val.register(ops.StringSQLLike)
def _string_like(op, *, arg, pattern, **_):
    return arg.like(pattern)


@translate_val.register(ops.StringSQLILike)
def _string_ilike(op, *, arg, pattern, **_):
    return arg.ilike(pattern)


@translate_val.register(ops.Capitalize)
def _string_capitalize(op, *, arg, **_):
    return sg.exp.Concat(
        expressions=[f.upper(f.substr(arg, 1, 1)), f.lower(f.substr(arg, 2))]
    )


### NULL PLAYER CHARACTER
@translate_val.register(ops.IsNull)
def _is_null(op, *, arg, **_):
    return arg.is_(NULL)


@translate_val.register(ops.NotNull)
def _is_not_null(op, *, arg, **_):
    return sg.not_(arg.is_(NULL))


### Definitely Not Tensors


@translate_val.register(ops.ArrayDistinct)
def _array_sort(op, *, arg, **_):
    return if_(
        arg.is_(NULL),
        NULL,
        f.list_distinct(arg)
        + if_(f.list_count(arg) < f.len(arg), f.array(NULL), f.array()),
    )


@translate_val.register(ops.ArrayIndex)
def _array_index_op(op, *, arg, index, **_):
    return f.list_extract(arg, index + cast(index >= 0, op.index.dtype))


@translate_val.register(ops.InValues)
def _in_values(op, *, value, options, **_):
    return value.isin(*options)


@translate_val.register(ops.ArrayConcat)
def _array_concat(op, *, arg, **_):
    result, *rest = arg
    for arg in rest:
        result = f.list_concat(result, arg)
    return result


@translate_val.register(ops.ArrayRepeat)
def _array_repeat_op(op, *, arg, times, **_):
    return f.flatten(
        sg.select(f.array(sg.select(arg).from_(f.range(times)))).subquery()
    )


def _neg_idx_to_pos(array, idx):
    arg_length = f.len(array)
    return if_(
        idx >= 0,
        idx,
        # Need to have the greatest here to handle the case where
        # abs(neg_index) > arg_length
        # e.g. where the magnitude of the negative index is greater than the
        # length of the array
        # You cannot index a[:-3] if a = [1, 2]
        arg_length + f.greatest(idx, -arg_length),
    )


@translate_val.register(ops.ArraySlice)
def _array_slice_op(op, *, arg, start, stop, **_):
    arg_length = f.len(arg)

    if start is None:
        start = 0
    else:
        start = f.least(arg_length, _neg_idx_to_pos(arg, start))

    if stop is None:
        stop = arg_length
    else:
        stop = _neg_idx_to_pos(arg, stop)

    return f.list_slice(arg, start + 1, stop)


@translate_val.register(ops.ArrayStringJoin)
def _array_string_join(op, *, sep, arg, **_):
    return f.array_to_string(arg, sep)


@translate_val.register(ops.ArrayMap)
def _array_map(op, *, arg, body, param, **_):
    lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
    return f.list_apply(arg, lamduh)


@translate_val.register(ops.ArrayFilter)
def _array_filter(op, *, arg, body, param, **_):
    lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
    return f.list_filter(arg, lamduh)


@translate_val.register(ops.ArrayIntersect)
def _array_intersect(op, *, left, right, **_):
    param = sg.to_identifier("x")
    body = f.list_contains(right, param)
    lamduh = sg.exp.Lambda(this=body, expressions=[param])
    return f.list_filter(left, lamduh)


@translate_val.register(ops.ArrayRemove)
def _array_remove(op, *, arg, other, **_):
    param = sg.to_identifier("x")
    body = param.neq(other)
    lamduh = sg.exp.Lambda(this=body, expressions=[param])
    return f.list_filter(arg, lamduh)


@translate_val.register(ops.ArrayUnion)
def _array_union(op, *, left, right, **_):
    arg = f.list_concat(left, right)
    return if_(
        arg.is_(NULL),
        NULL,
        f.list_distinct(arg)
        + if_(f.list_count(arg) < f.len(arg), f.array(NULL), f.array()),
    )


@translate_val.register(ops.ArrayZip)
def _array_zip(op: ops.ArrayZip, *, arg, **_) -> str:
    i = sg.to_identifier("i")
    body = sg.exp.Struct.from_arg_list(
        [
            sg.exp.Slice(this=k, expression=v[i])
            for k, v in zip(map(sg.exp.convert, op.dtype.value_type.names), arg)
        ]
    )
    func = sg.exp.Lambda(this=body, expressions=[i])
    return f.list_apply(
        f.range(
            1,
            # DuckDB Range excludes upper bound
            f.greatest(*map(f.len, arg)) + 1,
        ),
        func,
    )


### Counting


@translate_val.register(ops.CountDistinct)
def _count_distinct(op, *, arg, where, **_):
    return agg.count(sg.exp.Distinct(expressions=[arg]), where=where)


@translate_val.register(ops.CountDistinctStar)
def _count_distinct_star(op, *, arg, where, **_):
    # use a tuple because duckdb doesn't accept COUNT(DISTINCT a, b, c, ...)
    #
    # this turns the expression into COUNT(DISTINCT (a, b, c, ...))
    row = sg.exp.Tuple(expressions=list(map(sg.column, op.arg.schema.keys())))
    return agg.count(sg.exp.Distinct(expressions=[row]), where=where)


@translate_val.register(ops.CountStar)
def _count_star(op, *, where, **_):
    return agg.count(STAR, where=where)


@translate_val.register(ops.Sum)
def _sum(op, *, arg, where, **_):
    arg = cast(arg, op.dtype) if op.arg.dtype.is_boolean() else arg
    return agg.sum(arg, where=where)


@translate_val.register(ops.NthValue)
def _nth_value(op, *, arg, nth, **_):
    return f.nth_value(arg, nth)


### Stats


@translate_val.register(ops.Quantile)
@translate_val.register(ops.MultiQuantile)
def _quantiles(op, *, arg, quantile, where, **_):
    return agg.quantile_cont(arg, quantile, where=where)


@translate_val.register(ops.Correlation)
def _corr(op, *, left, right, how, where, **_):
    if how == "sample":
        raise com.UnsupportedOperationError(
            "DuckDB only implements `pop` correlation coefficient"
        )

    # TODO: rewrite rule?
    if (left_type := op.left.dtype).is_boolean():
        left = cast(left, dt.Int32(nullable=left_type.nullable))

    if (right_type := op.right.dtype).is_boolean():
        right = cast(right, dt.Int32(nullable=right_type.nullable))

    return agg.corr(left, right, where=where)


@translate_val.register(ops.Covariance)
def _covariance(op, *, left, right, how, where, **_):
    _how = {"sample": "samp", "pop": "pop"}

    # TODO: rewrite rule?
    if (left_type := op.left.dtype).is_boolean():
        left = cast(left, dt.Int32(nullable=left_type.nullable))

    if (right_type := op.right.dtype).is_boolean():
        right = cast(right, dt.Int32(nullable=right_type.nullable))

    funcname = f"covar_{_how[how]}"
    return agg[funcname](left, right, where=where)


@translate_val.register(ops.Variance)
@translate_val.register(ops.StandardDev)
def _variance(op, *, arg, how, where, **_):
    _how = {"sample": "samp", "pop": "pop"}
    _func = {ops.Variance: "var", ops.StandardDev: "stddev"}

    if (arg_dtype := op.arg.dtype).is_boolean():
        arg = cast(arg, dt.Int32(nullable=arg_dtype.nullable))

    funcname = f"{_func[type(op)]}_{_how[how]}"
    return agg[funcname](arg, where=where)


@translate_val.register(ops.Arbitrary)
def _arbitrary(op, *, arg, how, where, **_):
    if how == "heavy":
        raise com.UnsupportedOperationError("how='heavy' not supported in the backend")
    funcs = {"first": agg.first, "last": agg.last}
    return funcs[how](arg, where=where)


@translate_val.register(ops.FindInSet)
def _index_of(op, *, needle, values, **_):
    return f.list_indexof(f.array(*values), needle)


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, *, base=None, cases, results, default, **_):
    return sg.exp.Case(this=base, ifs=list(map(if_, cases, results)), default=default)


@translate_val.register(ops.TableArrayView)
def _table_array_view(op, *, table, **_):
    return table.args["this"].subquery()


@translate_val.register(ops.ExistsSubquery)
def _exists_subquery(op, *, rel, **_):
    return f.exists(rel.this.subquery())


@translate_val.register(ops.InSubquery)
def _in_subquery(op, *, rel, needle, **_):
    return needle.isin(rel.this.subquery())


@translate_val.register(ops.ArrayColumn)
def _array_column(op, *, cols, **_):
    return f.array(*cols)


@translate_val.register(ops.StructColumn)
def _struct_column(op, *, names, values, **_):
    return sg.exp.Struct.from_arg_list(
        [
            sg.exp.Slice(this=sg.exp.convert(name), expression=value)
            for name, value in zip(names, values)
        ]
    )


@translate_val.register(ops.StructField)
def _struct_field(op, *, arg, field, **_):
    val = arg.this if isinstance(op.arg, ops.Alias) else arg
    return val[sg.exp.convert(field)]


@translate_val.register(ops.IdenticalTo)
def _identical_to(op, *, left, right, **_):
    return sg.exp.NullSafeEQ(this=left, expression=right)


@translate_val.register(ops.Greatest)
@translate_val.register(ops.Least)
@translate_val.register(ops.Coalesce)
def _vararg_func(op, *, arg, **_):
    return f[op.__class__.__name__.lower()](*arg)


@translate_val.register(ops.MapGet)
def _map_get(op, *, arg, key, default, **_):
    return f.ifnull(f.list_extract(f.element_at(arg, key), 1), default)


@translate_val.register(ops.MapContains)
def _map_contains(op, *, arg, key, **_):
    return f.len(f.element_at(arg, key)).neq(0)


def _binary_infix(sg_expr: sg.exp._Expression):
    def formatter(op, *, left, right, **_):
        return sg.exp.Paren(this=sg_expr(this=left, expression=right))

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
    ops.Xor: sg.exp.Xor,
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


### Ordering and window functions


@translate_val.register(ops.RowNumber)
def _row_number(op, **_):
    return sg.exp.RowNumber()


@translate_val.register(ops.DenseRank)
def _dense_rank(op, **_):
    return f.dense_rank()


@translate_val.register(ops.MinRank)
def _rank(op, **_):
    return f.rank()


@translate_val.register(ops.PercentRank)
def _percent_rank(op, **_):
    return f.percent_rank()


@translate_val.register(ops.CumeDist)
def _cume_dist(op, **_):
    return f.cume_dist()


@translate_val.register
def _sort_key(op: ops.SortKey, *, expr, ascending: bool, **_):
    return sg.exp.Ordered(this=expr, desc=not ascending)


@translate_val.register(ops.ApproxMedian)
def _approx_median(op, *, arg, where, **_):
    return agg.approx_quantile(arg, 0.5, where=where)


@translate_val.register(ops.WindowBoundary)
def _window_boundary(op, *, value, preceding, **_):
    # TODO: bit of a hack to return a dict, but there's no sqlglot expression
    # that corresponds to _only_ this information
    return {"value": value, "side": "preceding" if preceding else "following"}


@translate_val.register(ops.WindowFrame)
def _window_frame(op, *, group_by, order_by, start, end, **_):
    if start is None:
        start = {}

    start_value = start.get("value", "UNBOUNDED")
    start_side = start.get("side", "PRECEDING")

    if end is None:
        end = {}

    end_value = end.get("value", "UNBOUNDED")
    end_side = end.get("side", "FOLLOWING")

    spec = sg.exp.WindowSpec(
        kind=op.how.upper(),
        start=start_value,
        start_side=start_side,
        end=end_value,
        end_side=end_side,
        over="OVER",
    )

    order = sg.exp.Order(expressions=order_by) if order_by else None

    # TODO: bit of a hack to return a partial, but similar to `WindowBoundary`
    # there's no sqlglot expression that corresponds to _only_ this information
    return partial(sg.exp.Window, partition_by=group_by, order=order, spec=spec)


@translate_val.register(ops.WindowFunction)
def _window(op: ops.WindowFunction, *, func, frame, **_: Any):
    return frame(this=func)


@translate_val.register(ops.Lag)
@translate_val.register(ops.Lead)
def formatter(op, *, arg, offset, default, **_):
    args = [arg]

    if default is not None:
        if offset is None:
            offset = 1

        args.append(offset)
        args.append(default)
    elif offset is not None:
        args.append(offset)

    return f[type(op).__name__.lower()](*args)


@translate_val.register(ops.Argument)
def _argument(op, **_):
    return sg.to_identifier(op.name)


@translate_val.register(ops.RowID)
def _rowid(op, *, table, **_) -> str:
    return sg.column(op.name, table=table.alias_or_name)


@translate_val.register(ops.ScalarUDF)
def _scalar_udf(op, **kw) -> str:
    funcname = op.__full_name__
    return f[funcname](*kw.values())


@translate_val.register(ops.AggUDF)
def _agg_udf(op, *, where, **kw) -> str:
    return agg[op.__full_name__](*kw.values(), where=where)


@translate_val.register(ops.ToJSONMap)
@translate_val.register(ops.ToJSONArray)
def _to_json_collection(op, *, arg, **_):
    return f.try_cast(arg, DuckDBType.from_ibis(op.dtype))


@translate_val.register(ops.TimestampDelta)
@translate_val.register(ops.DateDelta)
@translate_val.register(ops.TimeDelta)
def _temporal_delta(op, *, part, left, right, **_):
    # dialect is necessary due to sqlglot's default behavior
    # of `part` coming last
    return f.date_diff(part, right, left, dialect="duckdb")


@translate_val.register(ops.TimestampBucket)
def _timestamp_bucket(op, *, arg, interval, offset, **_):
    origin = f.cast("epoch", DuckDBType.from_ibis(dt.timestamp))
    if offset is not None:
        origin += offset
    return f.time_bucket(interval, arg, origin)
