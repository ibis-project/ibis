from __future__ import annotations

import functools
import operator
from typing import Any

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import (
    NULL,
    STAR,
    AggGen,
    F,
    interval,
    make_cast,
    paren,
    parenthesize,
)
from ibis.backends.base.sqlglot.datatypes import PostgresType
from ibis.common.temporal import IntervalUnit
from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowType


def _aggregate(funcname, *args, where):
    expr = F[funcname](*args)
    if where is not None:
        return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
    return expr


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


agg = AggGen(aggfunc=_aggregate)
cast = make_cast(PostgresType)
if_ = F["if"]

_simple_ops = {
    ops.Abs: "abs",
    ops.Ln: "ln",
    ops.Log2: "log2",
    ops.Log10: "log10",
    ops.Sqrt: "sqrt",
    ops.Reverse: "reverse",
    ops.Strip: "trim",
    ops.LStrip: "ltrim",
    ops.RStrip: "rtrim",
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.StringLength: "character_length",
    ops.Capitalize: "initcap",
    ops.Repeat: "repeat",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.Count: "count",
    ops.Min: "min",
    ops.Max: "max",
    ops.Mean: "avg",
    ops.Median: "median",
    ops.ApproxMedian: "approx_median",
    ops.Acos: "acos",
    ops.Asin: "asin",
    ops.Atan: "atan",
    ops.Cos: "cos",
    ops.Sin: "sin",
    ops.Tan: "tan",
    ops.Exp: "exp",
    ops.Power: "power",
    ops.RandomScalar: "random",
    ops.Translate: "translate",
    ops.StringAscii: "ascii",
    ops.StartsWith: "starts_with",
    ops.StrRight: "right",
    ops.StringReplace: "replace",
    ops.Sign: "sign",
    ops.ExtractMicrosecond: "extract_microsecond",
    ops.RowNumber: "row_number",
    ops.Any: "bool_or",
    ops.All: "bool_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.BitAnd: "bit_and",
    ops.ApproxCountDistinct: "approx_distinct",
    ops.BitwiseAnd: "bit_and",
    ops.Lag: "lag",
    ops.Lead: "lead",
    ops.First: "first_value",
    ops.Last: "last_value",
    ops.DenseRank: "dense_rank",
    ops.PercentRank: "percent_rank",
    ops.NTile: "ntile",
    ops.MinRank: "rank",
    ops.CumeDist: "cume_dist",
    ops.NthValue: "nth_value",
    ops.Cot: "cot",
    ops.Atan2: "atan2",
    ops.Radians: "radians",
    ops.Degrees: "degrees",
    ops.NullIf: "nullif",
    ops.Pi: "pi",
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

_binary_infix_ops = {
    # Binary operations
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
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
}


def _binary_infix(func):
    def formatter(op, *, left, right, **_):
        return func(parenthesize(op.left, left), parenthesize(op.right, right))

    return formatter


for _op, _func in _binary_infix_ops.items():
    translate_val.register(_op)(_binary_infix(_func))

del _op, _func


@translate_val.register(ops.Alias)
def alias(op, *, arg, name, **_):
    return arg.as_(name)


def _to_timestamp(value, target_dtype, literal=False):
    tz = (
        f'Some("{timezone}")'
        if (timezone := target_dtype.timezone) is not None
        else "None"
    )
    unit = (
        target_dtype.unit.name.capitalize()
        if target_dtype.scale is not None
        else "Microsecond"
    )
    str_value = str(value) if literal else value
    return F.arrow_cast(str_value, f"Timestamp({unit}, {tz})")


@translate_val.register(ops.Literal)
def _literal(op, *, value, dtype, **kw):
    if value is None and dtype.nullable:
        if dtype.is_null():
            return NULL
        return cast(NULL, dtype)
    elif dtype.is_boolean():
        return sg.exp.convert(bool(value))
    elif dtype.is_inet():
        # treat inet as strings for now
        return sg.exp.convert(str(value))
    elif dtype.is_decimal():
        return cast(
            sg.exp.convert(str(value)),
            dt.Decimal(precision=dtype.precision or 38, scale=dtype.scale or 9),
        )
    elif dtype.is_string() or dtype.is_macaddr():
        return sg.exp.convert(str(value))
    elif dtype.is_numeric():
        return sg.exp.convert(value)
    elif dtype.is_interval():
        if dtype.unit.short in {"ms", "us", "ns"}:
            raise com.UnsupportedOperationError(
                "DataFusion doesn't support subsecond interval resolutions"
            )

        return interval(value, unit=dtype.resolution.upper())
    elif dtype.is_timestamp():
        return _to_timestamp(value, dtype, literal=True)
    elif dtype.is_date():
        return F.date_trunc("day", value.isoformat())
    elif dtype.is_time():
        return cast(sg.exp.convert(str(value)), dt.time())
    elif dtype.is_array():
        vtype = dtype.value_type
        values = [
            _literal(ops.Literal(v, dtype=vtype), value=v, dtype=vtype, **kw)
            for v in value
        ]
        return F.array(*values)
    elif dtype.is_map():
        vtype = dtype.value_type
        keys = []
        values = []

        for k, v in value.items():
            keys.append(sg.exp.convert(k))
            values.append(
                _literal(ops.Literal(v, dtype=vtype), value=v, dtype=vtype, **kw)
            )

        return F.map(F.array(*keys), F.array(*values))
    elif dtype.is_struct():
        fields = [
            _literal(ops.Literal(v, dtype=ftype), value=v, dtype=ftype, **kw)
            for ftype, v in zip(dtype.types, value.values())
        ]
        return cast(sg.exp.Struct.from_arg_list(fields), dtype)
    elif dtype.is_binary():
        return sg.exp.HexString(this=value.hex())
    else:
        raise NotImplementedError(f"Unsupported type: {dtype!r}")


@translate_val.register(ops.Cast)
def _cast(op, *, arg, to, **_):
    if to.is_interval():
        unit_name = to.unit.name.lower()
        return sg.cast(F.concat(sg.cast(arg, "text"), f" {unit_name}"), "interval")
    if to.is_timestamp():
        return _to_timestamp(arg, to)
    if to.is_decimal():
        return F.arrow_cast(arg, f"{PyArrowType.from_ibis(to)}".capitalize())
    return cast(arg, to)


@translate_val.register(ops.TableColumn)
def column(op, *, table, name, **_):
    return sg.column(name, table=table.alias_or_name, quoted=True)


@translate_val.register
def sort_key(op: ops.SortKey, *, expr, ascending: bool, **_):
    return sg.exp.Ordered(this=expr, desc=not ascending)


@translate_val.register(ops.Not)
def invert(op, *, arg, **_):
    if isinstance(arg, sg.exp.Filter):
        return sg.exp.Filter(
            this=_de_morgan_law(arg.this), expression=arg.expression
        )  # transform the not expression using _de_morgan_law
    return sg.not_(paren(arg))


def _de_morgan_law(logical_op: sg.exp.Expression):
    if isinstance(logical_op, sg.exp.LogicalAnd):
        return sg.exp.LogicalOr(this=sg.not_(paren(logical_op.this)))
    if isinstance(logical_op, sg.exp.LogicalOr):
        return sg.exp.LogicalAnd(this=sg.not_(paren(logical_op.this)))
    return None


@translate_val.register(ops.Ceil)
@translate_val.register(ops.Floor)
def ceil_floor(op, *, arg, **_):
    return cast(F[type(op).__name__.lower()](arg), dt.int64)


@translate_val.register(ops.Round)
def round(op, *, arg, digits, **_):
    if digits is not None:
        return F.round(arg, digits)
    return F.round(arg)


@translate_val.register(ops.Substring)
def substring(op, *, arg, start, length, **_):
    start += 1
    if length is not None:
        return F.substr(arg, start, length)
    return F.substr(arg, start)


@translate_val.register(ops.Divide)
def div(op, *, left, right, **_):
    return cast(left, dt.float64) / cast(right, dt.float64)


@translate_val.register(ops.FloorDivide)
def floordiv(op, *, left, right, **_):
    return F.floor(left / right)


@translate_val.register(ops.CountDistinct)
def count_distinct(op, *, arg, where, **_):
    return agg.count(sg.exp.Distinct(expressions=[arg]), where=where)


@translate_val.register(ops.CountStar)
def count_star(op, *, where, **_):
    return agg.count(STAR, where=where)


@translate_val.register(ops.Sum)
def sum(op, *, arg, where, **_):
    if op.arg.dtype.is_boolean():
        arg = cast(arg, dt.int64)
    return agg.sum(arg, where=where)


@translate_val.register(ops.Variance)
def variance(op, *, arg, how, where, **_):
    if how == "sample":
        return agg.var_samp(arg, where=where)
    elif how == "pop":
        return agg.var_pop(arg, where=where)
    else:
        raise ValueError(f"Unrecognized how value: {how}")


@translate_val.register(ops.StandardDev)
def stddev(op, *, arg, how, where, **_):
    if how == "sample":
        return agg.stddev_samp(arg, where=where)
    elif how == "pop":
        return agg.stddev_pop(arg, where=where)
    else:
        raise ValueError(f"Unrecognized how value: {how}")


@translate_val.register(ops.InValues)
def in_values(op, *, value, options, **_):
    return parenthesize(op.value, value).isin(*options)


@translate_val.register(ops.Negate)
def negate(op, *, arg, **_):
    return -paren(arg)


@translate_val.register(ops.Coalesce)
def coalesce(op, *, arg, **_):
    return F.coalesce(*arg)


@translate_val.register(ops.Log)
def log(op, *, arg, base, **_):
    return F.log(base, arg)


@translate_val.register(ops.E)
def e(op, **_):
    return F.exp(1)


@translate_val.register(ops.ScalarUDF)
def scalar_udf(op, **kw):
    input_type = op.__input_type__
    if input_type in (InputType.PYARROW, InputType.BUILTIN):
        return F[op.__full_name__](*kw.values())
    else:
        raise NotImplementedError(
            f"DataFusion only supports PyArrow UDFs: got a {input_type.name.lower()} UDF"
        )


@translate_val.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op, *, func, func_args, **_):
    return F[func.__name__](*func_args)


@translate_val.register(ops.AggUDF)
def agg_udf(op, *, where, **kw):
    return agg[op.__full_name__](*kw.values(), where=where)


@translate_val.register(ops.StringConcat)
def string_concat(op, *, arg, **_):
    return F.concat(*arg)


@translate_val.register(ops.RegexExtract)
def regex_extract(op, *, arg, pattern, index, **_):
    if not isinstance(op.index, ops.Literal):
        raise ValueError(
            "re_extract `index` expressions must be literals. "
            "Arbitrary expressions are not supported in the DataFusion backend"
        )
    return F.regexp_match(arg, F.concat("(", pattern, ")"))[index]


@translate_val.register(ops.RegexReplace)
def regex_replace(op, *, arg, pattern, replacement, **_):
    return F.regexp_replace(arg, pattern, replacement, sg.exp.convert("g"))


@translate_val.register(ops.StringFind)
def string_find(op, *, arg, substr, start, end, **_):
    if end is not None:
        raise NotImplementedError("`end` not yet implemented")

    if start is not None:
        pos = F.strpos(F.substr(arg, start + 1), substr)
        return F.coalesce(F.nullif(pos + start, start), 0)

    return F.strpos(arg, substr)


@translate_val.register(ops.RegexSearch)
def regex_search(op, *, arg, pattern, **_):
    return F.array_length(F.regexp_match(arg, pattern)) > 0


@translate_val.register(ops.StringContains)
def string_contains(op, *, haystack, needle, **_):
    return F.strpos(haystack, needle) > sg.exp.convert(0)


@translate_val.register(ops.StringJoin)
def string_join(op, *, sep, arg, **_):
    if not isinstance(op.sep, ops.Literal):
        raise ValueError(
            "join `sep` expressions must be literals. "
            "Arbitrary expressions are not supported in the DataFusion backend"
        )

    return F.concat_ws(sep, *arg)


@translate_val.register(ops.ExtractFragment)
def _(op, *, arg, **_):
    return F.extract_url_field(arg, "fragment")


@translate_val.register(ops.ExtractProtocol)
def extract_protocol(op, *, arg, **_):
    return F.extract_url_field(arg, "scheme")


@translate_val.register(ops.ExtractAuthority)
def extract_authority(op, *, arg, **_):
    return F.extract_url_field(arg, "netloc")


@translate_val.register(ops.ExtractPath)
def extract_path(op, *, arg, **_):
    return F.extract_url_field(arg, "path")


@translate_val.register(ops.ExtractHost)
def extract_host(op, *, arg, **_):
    return F.extract_url_field(arg, "hostname")


@translate_val.register(ops.ExtractQuery)
def extract_query(op, *, arg, key, **_):
    if key is not None:
        return F.extract_query_param(arg, key)
    return F.extract_query(arg)


@translate_val.register(ops.ExtractUserInfo)
def extract_user_info(op, *, arg, **_):
    return F.extract_user_info(arg)


@translate_val.register(ops.ExtractYear)
@translate_val.register(ops.ExtractMonth)
@translate_val.register(ops.ExtractQuarter)
@translate_val.register(ops.ExtractDay)
def extract(op, *, arg, **_):
    skip = len("Extract")
    part = type(op).__name__[skip:].lower()
    return F.date_part(part, arg)


@translate_val.register(ops.ExtractDayOfYear)
def extract_day_of_the_year(op, *, arg, **_):
    return F.date_part("doy", arg)


@translate_val.register(ops.DayOfWeekIndex)
def extract_day_of_the_week_index(op, *, arg, **_):
    return (F.date_part("dow", arg) + 6) % 7


_DOW_INDEX_NAME = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


@translate_val.register(ops.DayOfWeekName)
def extract_day_of_the_week_name(op, *, arg, **_):
    cases, results = zip(*_DOW_INDEX_NAME.items())

    return sg.exp.Case(
        this=paren((F.date_part("dow", arg) + 6) % 7),
        ifs=list(map(if_, cases, results)),
    )


@translate_val.register(ops.Date)
def date(op, *, arg, **_):
    return F.date_trunc("day", arg)


@translate_val.register(ops.ExtractWeekOfYear)
def extract_week_of_year(op, *, arg, **_):
    return F.date_part("week", arg)


@translate_val.register(ops.TimestampTruncate)
def timestamp_truncate(op, *, arg, **_):
    unit = op.unit
    if unit in (
        IntervalUnit.MILLISECOND,
        IntervalUnit.MICROSECOND,
        IntervalUnit.NANOSECOND,
    ):
        raise com.UnsupportedOperationError(
            f"The function is not defined for time unit {unit}"
        )

    return F.date_trunc(unit.name.lower(), arg)


@translate_val.register(ops.ExtractEpochSeconds)
def extract_epoch_seconds(op, *, arg, **_):
    if op.arg.dtype.is_date():
        return F.extract_epoch_seconds_date(arg)
    elif op.arg.dtype.is_timestamp():
        return F.extract_epoch_seconds_timestamp(arg)
    else:
        raise com.OperationNotDefinedError(
            f"The function is not defined for {op.arg.dtype}"
        )


@translate_val.register(ops.ExtractMinute)
def extract_minute(op, *, arg, **_):
    if op.arg.dtype.is_date():
        return F.date_part("minute", arg)
    elif op.arg.dtype.is_time():
        return F.extract_minute_time(arg)
    elif op.arg.dtype.is_timestamp():
        return F.extract_minute_timestamp(arg)
    else:
        raise com.OperationNotDefinedError(
            f"The function is not defined for {op.arg.dtype}"
        )


@translate_val.register(ops.ExtractMillisecond)
def extract_millisecond(op, *, arg, **_):
    if op.arg.dtype.is_time():
        return F.extract_millisecond_time(arg)
    elif op.arg.dtype.is_timestamp():
        return F.extract_millisecond_timestamp(arg)
    else:
        raise com.OperationNotDefinedError(
            f"The function is not defined for {op.arg.dtype}"
        )


@translate_val.register(ops.ExtractHour)
def extract_hour(op, *, arg, **_):
    if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
        return F.date_part("hour", arg)
    elif op.arg.dtype.is_time():
        return F.extract_hour_time(arg)
    else:
        raise com.OperationNotDefinedError(
            f"The function is not defined for {op.arg.dtype}"
        )


@translate_val.register(ops.ExtractSecond)
def extract_second(op, *, arg, **_):
    if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
        return F.extract_second_timestamp(arg)
    elif op.arg.dtype.is_time():
        return F.extract_second_time(arg)
    else:
        raise com.OperationNotDefinedError(
            f"The function is not defined for {op.arg.dtype}"
        )


@translate_val.register(ops.TableArrayView)
def _table_array_view(op, *, table, **_):
    return table.args["this"].subquery()


@translate_val.register(ops.BitwiseAnd)
def _bitwise_and(op, *, left, right, **_):
    return sg.exp.BitwiseAnd(this=left, expression=right)


@translate_val.register(ops.BitwiseOr)
def _bitwise_and(op, *, left, right, **_):
    return sg.exp.BitwiseOr(this=left, expression=right)


@translate_val.register(ops.BitwiseXor)
def _bitwise_and(op, *, left, right, **_):
    return sg.exp.BitwiseXor(this=left, expression=right)


@translate_val.register(ops.BitwiseLeftShift)
def _bitwise_and(op, *, left, right, **_):
    return sg.exp.BitwiseLeftShift(this=left, expression=right)


@translate_val.register(ops.BitwiseRightShift)
def _bitwise_and(op, *, left, right, **_):
    return sg.exp.BitwiseRightShift(this=left, expression=right)


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
    return functools.partial(
        sg.exp.Window, partition_by=group_by, order=order, spec=spec
    )


@translate_val.register(ops.WindowFunction)
def _window(op: ops.WindowFunction, *, func, frame, **_: Any):
    # frame is a partial call to sg.exp.Window
    return frame(this=func)


@translate_val.register(ops.WindowBoundary)
def _window_boundary(op, *, value, preceding, **_):
    # TODO: bit of a hack to return a dict, but there's no sqlglot expression
    # that corresponds to _only_ this information
    return {"value": value, "side": "preceding" if preceding else "following"}


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, *, base=None, cases, results, default, **_):
    return sg.exp.Case(this=base, ifs=list(map(if_, cases, results)), default=default)


@translate_val.register(ops.IfElse)
def _if_else(op, *, bool_expr, true_expr, false_null_expr, **_):
    return if_(bool_expr, true_expr, false_null_expr)


@translate_val.register(ops.NotNull)
def _not_null(op, *, arg, **_):
    return sg.not_(arg.is_(NULL))
