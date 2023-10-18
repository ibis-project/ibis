from __future__ import annotations

import calendar
import functools
import math
import operator
from functools import partial
from typing import Any

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot import NULL, STAR, AggGen, C, F, interval, make_cast
from ibis.backends.clickhouse.datatypes import ClickhouseType


def _aggregate(funcname, *args, where):
    has_filter = where is not None
    func = F[funcname + "If" * has_filter]
    args += (where,) * has_filter
    return func(*args)


agg = AggGen(aggfunc=_aggregate)
if_ = F["if"]
cast = make_cast(ClickhouseType)


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_val.register(ops.TableColumn)
def _column(op, *, table, name, **_):
    return sg.column(name, table=table.alias_or_name)


@translate_val.register(ops.Alias)
def _alias(op, *, arg, name, **_):
    return arg.as_(name)


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
def _cast(op, *, arg, to, **_):
    if to.is_interval():
        suffix = _interval_cast_suffixes[to.unit.short]
        return F[f"toInterval{suffix}"](arg)

    result = cast(arg, to)
    if (timezone := getattr(to, "timezone", None)) is not None:
        return F.toTimeZone(result, timezone)
    return result


@translate_val.register(ops.TryCast)
def _try_cast(op, *, arg, to, **_):
    return F.accurateCastOrNull(arg, ClickhouseType.to_string(to))


@translate_val.register(ops.Between)
def _between(op, *, arg, lower_bound, upper_bound, **_):
    return sg.exp.Between(this=arg, low=lower_bound, high=upper_bound)


@translate_val.register(ops.Negate)
def _negate(op, *, arg, **_):
    return -sg.exp.Paren(this=arg)


@translate_val.register(ops.Not)
def _not(op, *, arg, **_):
    return sg.not_(sg.exp.Paren(this=arg))


def _parenthesize(op, arg):
    if isinstance(op, (ops.Binary, ops.Unary)):
        return sg.exp.Paren(this=arg)
    else:
        # function calls don't need parens
        return arg


@translate_val.register(ops.ArrayIndex)
def _array_index_op(op, *, arg, index, **_):
    return arg[if_(index >= 0, index + 1, index)]


@translate_val.register(ops.ArrayRepeat)
def _array_repeat_op(op, *, arg, times, **_):
    return (
        sg.select(F.arrayFlatten(F.groupArray(C.arr)))
        .from_(
            sg.select(arg.as_("arr"))
            .from_(sg.table("numbers", db="system"))
            .limit(times)
            .subquery()
        )
        .subquery()
    )


@translate_val.register(ops.ArraySlice)
def _array_slice_op(op, *, arg, start, stop, **_):
    start = _parenthesize(op.start, start)
    start_correct = if_(start < 0, start, start + 1)

    if stop is not None:
        stop = _parenthesize(op.stop, stop)

        length = if_(
            stop < 0,
            stop,
            if_(
                start < 0,
                F.greatest(0, stop - (F.length(arg) + start)),
                F.greatest(0, stop - start),
            ),
        )
        return F.arraySlice(arg, start_correct, length)
    else:
        return F.arraySlice(arg, start_correct)


@translate_val.register(ops.CountStar)
def _count_star(op, *, where, **_):
    if where is not None:
        return F.countIf(where)
    return sg.exp.Count(this=STAR)


def _quantile(func: str):
    def _compile(op, *, arg, quantile, where, **_):
        if where is None:
            return agg.quantile(arg, quantile, where=where)

        return sg.exp.ParameterizedAgg(
            this=f"{func}If",
            expressions=util.promote_list(quantile),
            params=[arg, where],
        )

    return _compile


translate_val.register(ops.Quantile)(_quantile("quantile"))
translate_val.register(ops.MultiQuantile)(_quantile("quantiles"))


def _agg_variance_like(func):
    variants = {"sample": f"{func}Samp", "pop": f"{func}Pop"}

    def formatter(_, *, how, where, **kw):
        funcname = variants[how]
        return agg[funcname](*kw.values(), where=where)

    return formatter


@translate_val.register(ops.Correlation)
def _corr(op, *, left, right, how, where, **_):
    if how == "pop":
        raise ValueError("ClickHouse only implements `sample` correlation coefficient")
    return agg.corr(left, right, where=where)


@translate_val.register(ops.Arbitrary)
def _arbitrary(op, *, arg, how, where, **_):
    if how == "first":
        return agg.any(arg, where=where)
    elif how == "last":
        return agg.anyLast(arg, where=where)
    else:
        assert how == "heavy"
        return agg.anyHeavy(arg, where=where)


@translate_val.register(ops.Substring)
def _substring(op, *, arg, start, length, **_):
    # Clickhouse is 1-indexed
    suffix = (length,) * (length is not None)
    if_pos = F.substring(arg, start + 1, *suffix)
    if_neg = F.substring(arg, F.length(arg) + start + 1, *suffix)
    return if_(start >= 0, if_pos, if_neg)


@translate_val.register(ops.StringFind)
def _string_find(op, *, arg, substr, start, end, **_):
    if end is not None:
        raise com.UnsupportedOperationError("String find doesn't support end argument")

    if start is not None:
        return F.locate(arg, substr, start)

    return F.locate(arg, substr)


@translate_val.register(ops.RegexSearch)
def _regex_search(op, *, arg, pattern, **_):
    return sg.exp.RegexpLike(this=arg, expression=pattern)


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, *, arg, pattern, index, **_):
    arg = cast(arg, dt.String(nullable=False))

    pattern = F.concat("(", pattern, ")")

    if index is None:
        index = 0

    index += 1

    then = F.extractGroups(arg, pattern)[index]

    return if_(F.notEmpty(then), then, NULL)


@translate_val.register(ops.FindInSet)
def _index_of(op, *, needle, values, **_):
    return F.indexOf(F.array(*values), needle)


@translate_val.register(ops.Round)
def _round(op, *, arg, digits, **_):
    if digits is not None:
        return F.round(arg, digits)
    return F.round(arg)


@translate_val.register(ops.Sign)
def _sign(op, *, arg, **_):
    """Workaround for missing sign function."""
    return F.intDivOrZero(arg, F.abs(arg))


@translate_val.register(ops.Hash)
def _hash(op, *, arg, **_):
    return F.sipHash64(arg)


_SUPPORTED_ALGORITHMS = frozenset(
    (
        "MD5",
        "halfMD5",
        "SHA1",
        "SHA224",
        "SHA256",
        "intHash32",
        "intHash64",
        "cityHash64",
        "sipHash64",
        "sipHash128",
    )
)


@translate_val.register(ops.HashBytes)
def _hash_bytes(op, *, arg, how, **_):
    if how not in _SUPPORTED_ALGORITHMS:
        raise com.UnsupportedOperationError(f"Unsupported hash algorithm {how}")

    return F[how](arg)


@translate_val.register(ops.Log)
def _log(op, *, arg, base, **_):
    if base is None:
        return F.ln(arg)
    elif str(base) in ("2", "10"):
        return F[f"log{base}"](arg)
    else:
        return F.ln(arg) / F.ln(base)


@translate_val.register(ops.IntervalFromInteger)
def _interval_from_integer(op, *, arg, unit, **_):
    dtype = op.dtype
    if dtype.unit.short in ("ms", "us", "ns"):
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    return interval(arg, unit=dtype.resolution.upper())


@translate_val.register(ops.Literal)
def _literal(op, *, value, dtype, **kw):
    if value is None and dtype.nullable:
        if dtype.is_null():
            return NULL
        return cast(NULL, dtype)
    elif dtype.is_boolean():
        return sg.exp.convert(bool(value))
    elif dtype.is_inet():
        v = str(value)
        return F.toIPv6(v) if ":" in v else F.toIPv4(v)
    elif dtype.is_string():
        return sg.exp.convert(str(value).replace(r"\0", r"\\0"))
    elif dtype.is_macaddr():
        return sg.exp.convert(str(value))
    elif dtype.is_decimal():
        precision = dtype.precision
        if precision is None or not 1 <= precision <= 76:
            raise NotImplementedError(
                f"Unsupported precision. Supported values: [1 : 76]. Current value: {precision!r}"
            )

        if 1 <= precision <= 9:
            type_name = F.toDecimal32
        elif 10 <= precision <= 18:
            type_name = F.toDecimal64
        elif 19 <= precision <= 38:
            type_name = F.toDecimal128
        else:
            type_name = F.toDecimal256
        return type_name(value, dtype.scale)
    elif dtype.is_numeric():
        if math.isnan(value):
            return sg.exp.Literal(this="NaN", is_string=False)
        elif math.isinf(value):
            inf = sg.exp.Literal(this="inf", is_string=False)
            return -inf if value < 0 else inf
        return sg.exp.convert(value)
    elif dtype.is_interval():
        if dtype.unit.short in ("ms", "us", "ns"):
            raise com.UnsupportedOperationError(
                "Clickhouse doesn't support subsecond interval resolutions"
            )

        return interval(value, unit=dtype.resolution.upper())
    elif dtype.is_timestamp():
        funcname = "toDateTime"
        fmt = "%Y-%m-%dT%H:%M:%S"

        if micros := value.microsecond:
            funcname += "64"
            fmt += ".%f"

        args = [value.strftime(fmt)]

        if micros % 1000:
            args.append(6)
        elif micros // 1000:
            args.append(3)

        if (timezone := dtype.timezone) is not None:
            args.append(timezone)

        return F[funcname](*args)
    elif dtype.is_date():
        return F.toDate(value.strftime("%Y-%m-%d"))
    elif dtype.is_array():
        value_type = dtype.value_type
        values = [
            _literal(ops.Literal(v, dtype=value_type), value=v, dtype=value_type, **kw)
            for v in value
        ]
        return F.array(*values)
    elif dtype.is_map():
        value_type = dtype.value_type
        keys = []
        values = []

        for k, v in value.items():
            keys.append(sg.exp.convert(k))
            values.append(
                _literal(
                    ops.Literal(v, dtype=value_type), value=v, dtype=value_type, **kw
                )
            )

        return F.map(F.array(*keys), F.array(*values))
    elif dtype.is_struct():
        fields = [
            _literal(ops.Literal(v, dtype=field_type), value=v, dtype=field_type, **kw)
            for field_type, v in zip(dtype.types, value.values())
        ]
        return F.tuple(*fields)
    else:
        raise NotImplementedError(f"Unsupported type: {dtype!r}")


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, *, base=None, cases, results, default, **_):
    return sg.exp.Case(this=base, ifs=list(map(if_, cases, results)), default=default)


@translate_val.register(ops.TableArrayView)
def _table_array_view(op, *, table, **_):
    return table.args["this"].subquery()


@translate_val.register(ops.TimestampFromUNIX)
def _timestamp_from_unix(op, *, arg, unit, **_):
    if (unit := unit.short) in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")
    return F.toDateTime(arg)


@translate_val.register(ops.DateTruncate)
@translate_val.register(ops.TimestampTruncate)
@translate_val.register(ops.TimeTruncate)
def _truncate(op, *, arg, unit, **_):
    converters = {
        "Y": F.toStartOfYear,
        "M": F.toStartOfMonth,
        "W": F.toMonday,
        "D": F.toDate,
        "h": F.toStartOfHour,
        "m": F.toStartOfMinute,
        "s": F.toDateTime,
    }

    unit = unit.short
    if (converter := converters.get(unit)) is None:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

    return converter(arg)


@translate_val.register(ops.TimestampBucket)
def _timestamp_bucket(op, *, arg, interval, offset, **_):
    if offset is not None:
        raise com.UnsupportedOperationError(
            "Timestamp bucket with offset is not supported"
        )

    return F.toStartOfInterval(arg, interval)


@translate_val.register(ops.DateFromYMD)
def _date_from_ymd(op, *, year, month, day, **_):
    return F.toDate(
        F.concat(
            F.toString(year),
            "-",
            F.leftPad(F.toString(month), 2, "0"),
            "-",
            F.leftPad(F.toString(day), 2, "0"),
        )
    )


@translate_val.register(ops.TimestampFromYMDHMS)
def _timestamp_from_ymdhms(op, *, year, month, day, hours, minutes, seconds, **_):
    to_datetime = F.toDateTime(
        F.concat(
            F.toString(year),
            "-",
            F.leftPad(F.toString(month), 2, "0"),
            "-",
            F.leftPad(F.toString(day), 2, "0"),
            " ",
            F.leftPad(F.toString(hours), 2, "0"),
            ":",
            F.leftPad(F.toString(minutes), 2, "0"),
            ":",
            F.leftPad(F.toString(seconds), 2, "0"),
        )
    )
    if timezone := op.dtype.timezone:
        return F.toTimeZone(to_datetime, timezone)
    return to_datetime


@translate_val.register(ops.ExistsSubquery)
def _exists_subquery(op, *, foreign_table, predicates, **_):
    # https://github.com/ClickHouse/ClickHouse/issues/6697
    #
    # this would work if clickhouse supported correlated subqueries
    subq = sg.select(1).from_(foreign_table).where(sg.condition(predicates)).subquery()
    return F.exists(subq)


@translate_val.register(ops.StringSplit)
def _string_split(op, *, arg, delimiter, **_):
    return F.splitByString(delimiter, cast(arg, dt.String(nullable=False)))


@translate_val.register(ops.StringJoin)
def _string_join(op, *, sep, arg, **_):
    return F.arrayStringConcat(F.array(*arg), sep)


@translate_val.register(ops.StringConcat)
def _string_concat(op, *, arg, **_):
    return F.concat(*arg)


@translate_val.register(ops.StringSQLLike)
def _string_like(op, *, arg, pattern, **_):
    return arg.like(pattern)


@translate_val.register(ops.StringSQLILike)
def _string_ilike(op, *, arg, pattern, **_):
    return arg.ilike(pattern)


@translate_val.register(ops.Capitalize)
def _string_capitalize(op, *, arg, **_):
    return F.concat(F.upper(F.substr(arg, 1, 1)), F.lower(F.substr(arg, 2)))


@translate_val.register(ops.GroupConcat)
def _group_concat(op, *, arg, sep, where, **_):
    call = agg.groupArray(arg, where=where)
    return if_(F.empty(call), NULL, F.arrayStringConcat(call, sep))


@translate_val.register(ops.StrRight)
def _string_right(op, *, arg, nchars, **_):
    nchars = _parenthesize(op.nchars, nchars)
    return F.substring(arg, -nchars)


@translate_val.register(ops.Cot)
def _cotangent(op, *, arg, **_):
    return 1.0 / F.tan(arg)


def _bit_agg(func: str):
    def _translate(op, *, arg, where, **_):
        if not (dtype := op.arg.dtype).is_unsigned_integer():
            nbits = dtype.nbytes * 8
            arg = F[f"reinterpretAsUInt{nbits}"](arg)
        return agg[func](arg, where=where)

    return _translate


@translate_val.register(ops.ArrayColumn)
def _array_column(op, *, cols, **_):
    return F.array(*cols)


@translate_val.register(ops.StructColumn)
def _struct_column(op, *, values, **_):
    # ClickHouse struct types cannot be nullable
    # (non-nested fields can be nullable)
    return cast(F.tuple(*values), op.dtype.copy(nullable=False))


@translate_val.register(ops.Clip)
def _clip(op, *, arg, lower, upper, **_):
    if upper is not None:
        arg = if_(F.isNull(arg), NULL, F.least(upper, arg))

    if lower is not None:
        arg = if_(F.isNull(arg), NULL, F.greatest(lower, arg))

    return arg


@translate_val.register(ops.StructField)
def _struct_field(op, *, arg, field: str, **_):
    arg_dtype = op.arg.dtype
    idx = arg_dtype.names.index(field)
    return cast(sg.exp.Dot(this=arg, expression=sg.exp.convert(idx + 1)), op.dtype)


@translate_val.register(ops.Repeat)
def _repeat(op, *, arg, times, **_):
    return F.repeat(arg, F.accurateCast(times, "UInt64"))


@translate_val.register(ops.FloorDivide)
def _floor_divide(op, *, left, right, **_):
    return F.floor(left / right)


@translate_val.register(ops.StringContains)
def _string_contains(op, haystack, needle, **_):
    return F.locate(haystack, needle) > 0


@translate_val.register(ops.InValues)
def _in_values(op, *, value, options, **_):
    return _parenthesize(op.value, value).isin(*options)


@translate_val.register(ops.InColumn)
def _in_column(op, *, value, options, **_):
    return value.isin(options.this if isinstance(options, sg.exp.Subquery) else options)


_DAYS = calendar.day_name
_NUM_WEEKDAYS = len(_DAYS)


@translate_val.register(ops.DayOfWeekIndex)
def _day_of_week_index(op, *, arg, **_):
    weekdays = _NUM_WEEKDAYS
    return (((F.toDayOfWeek(arg) - 1) % weekdays) + weekdays) % weekdays


@translate_val.register(ops.DayOfWeekName)
def day_of_week_name(op, *, arg, **_):
    # ClickHouse 20 doesn't support dateName
    #
    # ClickHouse 21 supports dateName is broken for regexen:
    # https://github.com/ClickHouse/ClickHouse/issues/32777
    #
    # ClickHouses 20 and 21 also have a broken case statement hence the ifnull:
    # https://github.com/ClickHouse/ClickHouse/issues/32849
    #
    # We test against 20 in CI, so we implement day_of_week_name as follows
    num_weekdays = _NUM_WEEKDAYS
    base = (((F.toDayOfWeek(arg) - 1) % num_weekdays) + num_weekdays) % num_weekdays
    return sg.exp.Case(
        this=base,
        ifs=[if_(i, day) for i, day in enumerate(_DAYS)],
        default=sg.exp.convert(""),
    )


@translate_val.register(ops.Greatest)
@translate_val.register(ops.Least)
@translate_val.register(ops.Coalesce)
def _vararg_func(op, *, arg, **_):
    return F[op.__class__.__name__.lower()](*arg)


@translate_val.register(ops.Map)
def _map(op, *, keys, values, **_):
    # cast here to allow lookups of nullable columns
    return cast(F.tuple(keys, values), op.dtype)


@translate_val.register(ops.MapGet)
def _map_get(op, *, arg, key, default, **_):
    return if_(F.mapContains(arg, key), arg[key], default)


@translate_val.register(ops.ArrayConcat)
def _array_concat(op, *, arg, **_):
    return F.arrayConcat(*arg)


def _binary_infix(func):
    def formatter(op, *, left, right, **_):
        left = _parenthesize(op.left, left)
        right = _parenthesize(op.right, right)
        return func(left, right)

    return formatter


_binary_infix_ops = {
    # Binary operations
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.Modulus: operator.mod,
    # Comparisons
    ops.Equals: sg.exp.Condition.eq,
    ops.NotEquals: sg.exp.Condition.neq,
    ops.GreaterEqual: operator.ge,
    ops.Greater: operator.gt,
    ops.LessEqual: operator.le,
    ops.Less: operator.lt,
    # Boolean comparisons
    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.Xor: F.xor,
    ops.DateAdd: operator.add,
    ops.DateSub: operator.sub,
    ops.DateDiff: operator.sub,
    ops.TimestampAdd: operator.add,
    ops.TimestampSub: operator.sub,
    ops.TimestampDiff: operator.sub,
}


for _op, _func in _binary_infix_ops.items():
    translate_val.register(_op)(_binary_infix(_func))

del _op, _func

translate_val.register(ops.BitAnd)(_bit_agg("groupBitAnd"))
translate_val.register(ops.BitOr)(_bit_agg("groupBitOr"))
translate_val.register(ops.BitXor)(_bit_agg("groupBitXor"))

translate_val.register(ops.StandardDev)(_agg_variance_like("stddev"))
translate_val.register(ops.Variance)(_agg_variance_like("var"))
translate_val.register(ops.Covariance)(_agg_variance_like("covar"))


_simple_ops = {
    ops.Power: "pow",
    # Unary operations
    ops.TypeOf: "toTypeName",
    ops.IsNan: "isNaN",
    ops.IsInf: "isInfinite",
    ops.Abs: "abs",
    ops.Ceil: "ceil",
    ops.Floor: "floor",
    ops.Exp: "exp",
    ops.Sqrt: "sqrt",
    ops.Ln: "log",
    ops.Log2: "log2",
    ops.Log10: "log10",
    ops.Acos: "acos",
    ops.Asin: "asin",
    ops.Atan: "atan",
    ops.Atan2: "atan2",
    ops.Cos: "cos",
    ops.Sin: "sin",
    ops.Tan: "tan",
    ops.Pi: "pi",
    ops.E: "e",
    ops.RandomScalar: "randCanonical",
    # Unary aggregates
    ops.ApproxMedian: "median",
    ops.Median: "quantileExactExclusive",
    # TODO: there is also a `uniq` function which is the
    #       recommended way to approximate cardinality
    ops.ApproxCountDistinct: "uniqHLL12",
    ops.Mean: "avg",
    ops.Sum: "sum",
    ops.Max: "max",
    ops.Min: "min",
    ops.Any: "max",
    ops.All: "min",
    ops.ArgMin: "argMin",
    ops.ArgMax: "argMax",
    ops.ArrayCollect: "groupArray",
    ops.Count: "count",
    ops.CountDistinct: "uniq",
    ops.First: "any",
    ops.Last: "anyLast",
    # string operations
    ops.StringLength: "length",
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.Reverse: "reverse",
    ops.StringReplace: "replaceAll",
    ops.StartsWith: "startsWith",
    ops.EndsWith: "endsWith",
    ops.LPad: "leftPad",
    ops.RPad: "rightPad",
    ops.LStrip: "trimLeft",
    ops.RStrip: "trimRight",
    ops.Strip: "trimBoth",
    ops.RegexReplace: "replaceRegexpAll",
    ops.StringAscii: "ascii",
    # Temporal operations
    ops.Date: "toDate",
    ops.TimestampNow: "now",
    ops.ExtractYear: "toYear",
    ops.ExtractMonth: "toMonth",
    ops.ExtractDay: "toDayOfMonth",
    ops.ExtractDayOfYear: "toDayOfYear",
    ops.ExtractQuarter: "toQuarter",
    ops.ExtractWeekOfYear: "toISOWeek",
    ops.ExtractHour: "toHour",
    ops.ExtractMinute: "toMinute",
    ops.ExtractSecond: "toSecond",
    # Other operations
    ops.E: "e",
    # for more than 2 args this should be arrayGreatest|Least(array([]))
    # because clickhouse"s greatest and least doesn"t support varargs
    ops.IfElse: "if",
    ops.ArrayLength: "length",
    ops.Unnest: "arrayJoin",
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.Strftime: "formatDateTime",
    ops.IsNull: "isNull",
    ops.NotNull: "isNotNull",
    ops.NullIf: "nullIf",
    ops.MapContains: "mapContains",
    ops.MapLength: "length",
    ops.MapKeys: "mapKeys",
    ops.MapValues: "mapValues",
    ops.MapMerge: "mapUpdate",
    ops.BitwiseAnd: "bitAnd",
    ops.BitwiseOr: "bitOr",
    ops.BitwiseXor: "bitXor",
    ops.BitwiseLeftShift: "bitShiftLeft",
    ops.BitwiseRightShift: "bitShiftRight",
    ops.BitwiseNot: "bitNot",
    ops.ArraySort: "arraySort",
    ops.ArrayContains: "has",
    ops.FirstValue: "first_value",
    ops.LastValue: "last_value",
    ops.NTile: "ntile",
    ops.ArrayIntersect: "arrayIntersect",
    ops.ExtractEpochSeconds: "toRelativeSecondNum",
    ops.NthValue: "nth_value",
    ops.MinRank: "rank",
    ops.DenseRank: "dense_rank",
    ops.RowNumber: "row_number",
    ops.ExtractProtocol: "protocol",
    ops.ExtractAuthority: "netloc",
    ops.ExtractHost: "domain",
    ops.ExtractPath: "path",
    ops.ExtractFragment: "fragment",
    ops.ArrayPosition: "indexOf",
}


for _op, _name in _simple_ops.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @translate_val.register(_op)
        def _fmt(_, _name: str = _name, *, where, **kw):
            return agg[_name](*kw.values(), where=where)

    else:

        @translate_val.register(_op)
        def _fmt(_, _name: str = _name, **kw):
            return F[_name](*kw.values())


del _fmt, _name, _op


@translate_val.register(ops.ArrayDistinct)
def _array_distinct(op, *, arg, **_):
    null_element = if_(F.countEqual(arg, NULL) > 0, F.array(NULL), F.array())
    return F.arrayConcat(F.arrayDistinct(arg), null_element)


@translate_val.register(ops.ExtractMicrosecond)
def _extract_microsecond(op, *, arg, **_):
    dtype = op.dtype
    return cast(
        F.toUnixTimestamp64Micro(cast(arg, op.arg.dtype.copy(scale=6))) % 1_000_000,
        dtype,
    )


@translate_val.register(ops.ExtractMillisecond)
def _extract_millisecond(op, *, arg, **_):
    dtype = op.dtype
    return cast(
        F.toUnixTimestamp64Milli(cast(arg, op.arg.dtype.copy(scale=3))) % 1_000, dtype
    )


@translate_val.register
def _sort_key(op: ops.SortKey, *, expr, ascending: bool, **_):
    return sg.exp.Ordered(this=expr, desc=not ascending)


@translate_val.register(ops.WindowBoundary)
def _window_boundary(op, *, value, preceding, **_):
    # TODO: bit of a hack to return a dict, but there's no sqlglot expression
    # that corresponds to _only_ this information
    return {"value": value, "side": "preceding" if preceding else "following"}


@translate_val.register(ops.RowsWindowFrame)
@translate_val.register(ops.RangeWindowFrame)
def _window_frame(op, *, group_by, order_by, start, end, max_lookback=None, **_):
    if max_lookback is not None:
        raise NotImplementedError(
            "`max_lookback` is not supported in the ClickHouse backend"
        )

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
    # frame is a partial call to sg.exp.Window
    return frame(this=func)


def shift_like(op_class, func):
    @translate_val.register(op_class)
    def formatter(op, *, arg, offset, default, **_):
        args = [arg]

        if default is not None:
            if offset is None:
                offset = 1

            args.append(offset)
            args.append(default)
        elif offset is not None:
            args.append(offset)

        return func(*args)

    return formatter


shift_like(ops.Lag, F.lagInFrame)
shift_like(ops.Lead, F.leadInFrame)


@translate_val.register(ops.ExtractFile)
def _extract_file(op, *, arg, **_):
    return F.cutFragment(F.pathFull(arg))


@translate_val.register(ops.ExtractQuery)
def _extract_query(op, *, arg, key, **_):
    if key is not None:
        return F.extractURLParameter(arg, key)
    else:
        return F.queryString(arg)


@translate_val.register(ops.ArrayStringJoin)
def _array_string_join(op, *, arg, sep, **_):
    return F.arrayStringConcat(arg, sep)


@translate_val.register(ops.Argument)
def _argument(op, *, name, **_):
    return sg.to_identifier(name)


@translate_val.register(ops.ArrayMap)
def _array_map(op, *, arg, param, body, **_):
    func = sg.exp.Lambda(this=body, expressions=[param])
    return F.arrayMap(func, arg)


@translate_val.register(ops.ArrayFilter)
def _array_filter(op, *, arg, param, body, **_):
    func = sg.exp.Lambda(this=body, expressions=[param])
    return F.arrayFilter(func, arg)


@translate_val.register(ops.ArrayRemove)
def _array_remove(op, *, arg, other, **_):
    x = sg.to_identifier("x")
    body = x.neq(other)
    return F.arrayFilter(sg.exp.Lambda(this=body, expressions=[x]), arg)


@translate_val.register(ops.ArrayUnion)
def _array_union(op, *, left, right, **_):
    arg = F.arrayConcat(left, right)
    null_element = if_(F.countEqual(arg, NULL) > 0, F.array(NULL), F.array())
    return F.arrayConcat(F.arrayDistinct(arg), null_element)


@translate_val.register(ops.ArrayZip)
def _array_zip(op: ops.ArrayZip, *, arg, **_: Any) -> str:
    return F.arrayZip(*arg)


@translate_val.register(ops.CountDistinctStar)
def _count_distinct_star(op: ops.CountDistinctStar, *, where, **_: Any) -> str:
    columns = F.tuple(*map(sg.column, op.arg.schema.names))

    if where is not None:
        return F.countDistinctIf(columns, where)
    else:
        return F.countDistinct(columns)


@translate_val.register(ops.ScalarUDF)
def _scalar_udf(op, **kw) -> str:
    return F[op.__full_name__](*kw.values())


@translate_val.register(ops.AggUDF)
def _agg_udf(op, *, where, **kw) -> str:
    return agg[op.__full_name__](*kw.values(), where=where)


@translate_val.register(ops.DateDelta)
@translate_val.register(ops.TimestampDelta)
def _delta(op, *, part, left, right, **_):
    return sg.exp.DateDiff(this=left, expression=right, unit=part)
