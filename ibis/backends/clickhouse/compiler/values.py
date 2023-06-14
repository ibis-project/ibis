from __future__ import annotations

import calendar
import contextlib
import functools
from functools import partial
from operator import add, mul, sub
from typing import Any, Literal, Mapping

import sqlglot as sg
from toolz import flip

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import helpers
from ibis.backends.clickhouse.datatypes import serialize

# TODO: Ideally we can translate bottom up a la `relations.py`
# TODO: Find a way to remove all the dialect="clickhouse" kwargs


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


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
        return sg.alias(val, op.name, dialect="clickhouse")
    return val


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
        return f"toInterval{suffix}({arg})"

    to = translate_val(op.to, **kw)
    result = f"CAST({arg} AS {to})"
    if (timezone := getattr(op.to, "timezone", None)) is not None:
        return f"toTimeZone({result}, {timezone!r})"
    return result


@translate_val.register(ops.TryCast)
def _try_cast(op, **kw):
    return f"accurateCastOrNull({translate_val(op.arg, **kw)}, '{serialize(op.to)}')"


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


@translate_val.register(ops.ExtractEpochSeconds)
def _extract_epoch_seconds(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"toRelativeSecondNum({arg})"


@translate_val.register(ops.ArrayIndex)
def _array_index_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    index = translate_val(op.index, **kw)
    correct_idx = f"if({index} >= 0, {index} + 1, {index})"
    return f"arrayElement({arg}, {correct_idx})"


@translate_val.register(ops.ArrayRepeat)
def _array_repeat_op(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    from_ = f"(SELECT {arg} AS arr FROM system.numbers LIMIT {times})"
    query = sg.parse_one(
        f"SELECT arrayFlatten(groupArray(arr)) FROM {from_}", read="clickhouse"
    )
    return query.subquery()


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


def _agg(func_name):
    def formatter(op, **kw):
        return _aggregate(op, func_name, where=op.where, **kw)

    return formatter


@translate_val.register(ops.CountStar)
def _count_star(op, **kw):
    if (where := op.where) is not None:
        return f"countIf({translate_val(where, **kw)})"
    return "count(*)"


@translate_val.register(ops.NotAny)
def _not_any(op, **kw):
    return translate_val(ops.All(ops.Not(op.arg), where=op.where), **kw)


@translate_val.register(ops.NotAll)
def _not_all(op, **kw):
    return translate_val(ops.Any(ops.Not(op.arg), where=op.where), **kw)


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
        raise TypeError("ClickHouse quantile only accepts a list of Python floats")

    quantile = ", ".join(map(str, op.quantile.value))
    return _quantile_like("quantiles", op, quantile, **kw)


def _agg_variance_like(func):
    variants = {"sample": f"{func}Samp", "pop": f"{func}Pop"}

    def formatter(op, **kw):
        return _aggregate(op, variants[op.how], where=op.where, **kw)

    return formatter


@translate_val.register(ops.Correlation)
def _corr(op, **kw):
    if op.how == "pop":
        raise ValueError("ClickHouse only implements `sample` correlation coefficient")
    return _aggregate(op, "corr", where=op.where, **kw)


def _aggregate(op, func, *, where=None, **kw):
    args = [
        translate_val(arg, **kw)
        for argname, arg in zip(op.argnames, op.args)
        if argname not in ("where", "how")
    ]
    if where is not None:
        func += "If"
        args.append(translate_val(where, **kw))
    elif func == "any":
        func = '"any"'

    joined_args = ", ".join(map(_sql, args))
    return f"{func}({joined_args})"


@translate_val.register(ops.Xor)
def _xor(op, **kw):
    raw_left = translate_val(op.left, **kw)
    left = _parenthesize(op.left, raw_left)
    raw_right = translate_val(op.right, **kw)
    right = _parenthesize(op.right, raw_right)
    return f"xor({left}, {right})"


@translate_val.register(ops.Arbitrary)
def _arbitrary(op, **kw):
    functions = {
        "first": "any",
        "last": "anyLast",
        "heavy": "anyHeavy",
    }
    return _aggregate(op, functions[op.how], where=op.where, **kw)


@translate_val.register(ops.Substring)
def _substring(op, **kw):
    # Clickhouse is 1-indexed
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

    if op.start is not None:
        op_start = translate_val(op.start)
        return f"locate({arg}, {substr}, {op_start}) - 1"

    return f"locate({arg}, {substr}) - 1"


@translate_val.register(ops.RegexSearch)
def _regex_search(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    return f"{arg} REGEXP {pattern}"


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, **kw):
    arg = translate_val(op.arg, **kw)
    arg = f"CAST({arg} AS String)"

    wrapped_op_pattern = op.pattern.copy(value="(" + op.pattern.value + ")")
    pattern = translate_val(wrapped_op_pattern, **kw)

    then = f"extractGroups({arg}, {pattern})[1]"
    if op.index is not None:
        index = translate_val(op.index, **kw)
        then = f"extractGroups({arg}, {pattern})[{index} + 1]"

    does_match = f"notEmpty({then})"

    return f"if({does_match}, {then}, NULL)"


@translate_val.register(ops.FindInSet)
def _index_of(op, **kw):
    values = map(partial(translate_val, **kw), op.values)
    values = ", ".join(map(_sql, values))
    needle = translate_val(op.needle, **kw)
    return f"indexOf([{values}], {needle}) - 1"


@translate_val.register(ops.Round)
def _round(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (digits := op.digits) is not None:
        return f"round({arg}, {translate_val(digits, **kw)})"
    return f"round({arg})"


@translate_val.register(ops.Sign)
def _sign(op, **kw):
    """Workaround for missing sign function."""
    arg = translate_val(op.arg, **kw)
    return f"intDivOrZero({arg}, abs({arg}))"


@translate_val.register(ops.Hash)
def _hash(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"sipHash64({arg})"


@translate_val.register(ops.HashBytes)
def _hash_bytes(op, **kw):
    algorithms = {
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
    }

    arg = translate_val(op.arg, **kw)
    if (how := op.how) not in algorithms:
        raise com.UnsupportedOperationError(f"Unsupported hash algorithm {how}")

    return f"{how}({arg})"


@translate_val.register(ops.Log)
def _log(op, **kw):
    arg = translate_val(op.arg, **kw)

    if has_base := (base := op.base) is not None:
        base = translate_val(base, **kw)

    func = "log"

    # base is translated at this point
    if has_base:
        if base != "2" and base != "10":
            raise ValueError(f"Base {base} for logarithm not supported!")
        else:
            func += base

    return f"{func}({arg})"


@translate_val.register(tuple)
def _node_list(op, punct="()", **kw):
    values = ", ".join(map(_sql, map(partial(translate_val, **kw), op)))
    return f"{punct[0]}{values}{punct[1]}"


def _interval_format(op):
    dtype = op.output_dtype
    if dtype.unit.short in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    return f"INTERVAL {op.value} {dtype.resolution.upper()}"


@translate_val.register(ops.IntervalFromInteger)
def _interval_from_integer(op, **kw):
    dtype = op.output_dtype
    if dtype.unit.short in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    arg = translate_val(op.arg, **kw)
    return f"INTERVAL {arg} {dtype.resolution.upper()}"


@translate_val.register(ops.Literal)
def _literal(op, **kw):
    value = op.value
    dtype = op.output_dtype
    if value is None and dtype.nullable:
        if dtype.is_null():
            return "Null"
        return f"CAST(Null AS {serialize(dtype)})"
    if dtype.is_boolean():
        return str(int(bool(value)))
    elif dtype.is_inet():
        v = str(value)
        return f"toIPv6({v!r})" if ":" in v else f"toIPv4({v!r})"
    elif dtype.is_string():
        quoted = value.replace("'", "''").replace("\\", "\\\\")
        return f"'{quoted}'"
    elif dtype.is_decimal():
        precision = dtype.precision
        if precision is None or not 1 <= precision <= 76:
            raise NotImplementedError(
                f'Unsupported precision. Supported values: [1 : 76]. Current value: {precision!r}'
            )

        if 1 <= precision <= 9:
            type_name = 'Decimal32'
        elif 10 <= precision <= 18:
            type_name = 'Decimal64'
        elif 19 <= precision <= 38:
            type_name = 'Decimal128'
        else:
            type_name = 'Decimal256'
        return f"to{type_name}({value!s}, {dtype.scale})"
    elif dtype.is_numeric():
        return repr(value)
    elif dtype.is_interval():
        return _interval_format(op)
    elif dtype.is_timestamp():
        func = "toDateTime"
        args = []

        fmt = "%Y-%m-%dT%H:%M:%S"

        if micros := value.microsecond:
            func = "toDateTime64"
            fmt += ".%f"

        args.append(value.strftime(fmt))
        if micros % 1000:
            args.append(6)
        elif micros // 1000:
            args.append(3)

        if (timezone := dtype.timezone) is not None:
            args.append(timezone)

        joined_args = ", ".join(map(repr, args))
        return f"{func}({joined_args})"

    elif dtype.is_date():
        formatted = value.strftime('%Y-%m-%d')
        return f"toDate('{formatted}')"
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
        raise NotImplementedError(f'Unsupported type: {dtype!r}')


def _sql(obj, dialect="clickhouse"):
    try:
        return obj.sql(dialect=dialect)
    except AttributeError:
        return obj


@translate_val.register(ops.SimpleCase)
@translate_val.register(ops.SearchedCase)
def _case(op, **kw):
    buf = ["CASE"]

    if (base := getattr(op, "base", None)) is not None:
        buf.append(translate_val(base, **kw))

    for when, then in zip(op.cases, op.results):
        buf.append(f"WHEN {translate_val(when, **kw)}")
        buf.append(f"THEN {translate_val(then, **kw)}")

    if (default := op.default) is not None:
        buf.append(f"ELSE {translate_val(default, **kw)}")

    buf.append("END")
    return " ".join(map(_sql, buf))


@translate_val.register(ops.TableArrayView)
def _table_array_view(op, *, cache, **kw):
    table = op.table
    try:
        return cache[table]
    except KeyError:
        from ibis.backends.clickhouse.compiler.relations import translate_rel

        # ignore the top level table, so that we can compile its dependencies
        (leaf,) = an.find_immediate_parent_tables(table, keep_input=False)
        res = translate_rel(table, table=cache[leaf], cache=cache, **kw)
        return res.subquery()


@translate_val.register(ops.TimestampFromUNIX)
def _timestamp_from_unix(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (unit := op.unit.short) in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    return f"toDateTime({arg})"


@translate_val.register(ops.DateTruncate)
@translate_val.register(ops.TimestampTruncate)
@translate_val.register(ops.TimeTruncate)
def _truncate(op, **kw):
    converters = {
        "Y": "toStartOfYear",
        "M": "toStartOfMonth",
        "W": "toMonday",
        "D": "toDate",
        "h": "toStartOfHour",
        "m": "toStartOfMinute",
        "s": "toDateTime",
    }

    unit = op.unit.short
    arg = translate_val(op.arg, **kw)
    try:
        converter = converters[unit]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

    return f"{converter}({arg})"


@translate_val.register(ops.DateFromYMD)
def _date_from_ymd(op, **kw):
    y = translate_val(op.year, **kw)
    m = translate_val(op.month, **kw)
    d = translate_val(op.day, **kw)
    return (
        "toDate(concat("
        f"toString({y}), '-', "
        f"leftPad(toString({m}), 2, '0'), '-', "
        f"leftPad(toString({d}), 2, '0')"
        "))"
    )


@translate_val.register(ops.TimestampFromYMDHMS)
def _timestamp_from_ymdhms(op, **kw):
    y = translate_val(op.year, **kw)
    m = translate_val(op.month, **kw)
    d = translate_val(op.day, **kw)
    h = translate_val(op.hours, **kw)
    min = translate_val(op.minutes, **kw)
    s = translate_val(op.seconds, **kw)

    to_datetime = (
        "toDateTime("
        f"concat(toString({y}), '-', "
        f"leftPad(toString({m}), 2, '0'), '-', "
        f"leftPad(toString({d}), 2, '0'), ' ', "
        f"leftPad(toString({h}), 2, '0'), ':', "
        f"leftPad(toString({min}), 2, '0'), ':', "
        f"leftPad(toString({s}), 2, '0')"
        "))"
    )
    if timezone := op.output_dtype.timezone:
        return f"toTimeZone({to_datetime}, {timezone})"
    return to_datetime


@translate_val.register(ops.ExistsSubquery)
@translate_val.register(ops.NotExistsSubquery)
def _exists_subquery(op, **kw):
    # https://github.com/ClickHouse/ClickHouse/issues/6697
    #
    # this would work, if clickhouse supported correlated subqueries
    from ibis.backends.clickhouse.compiler.relations import translate_rel

    foreign_table = translate_rel(op.foreign_table, **kw)
    predicates = translate_val(op.predicates, **kw)
    subq = (
        sg.select(1)
        .from_(foreign_table, dialect="clickhouse")
        .where(sg.condition(predicates), dialect="clickhouse")
    )
    prefix = "NOT " * isinstance(op, ops.NotExistsSubquery)
    return f"{prefix}EXISTS ({subq})"


@translate_val.register(ops.StringSplit)
def _string_split(op, **kw):
    arg = translate_val(op.arg, **kw)
    delimiter = translate_val(op.delimiter, **kw)
    return f"splitByString({delimiter}, CAST({arg} AS String))"


@translate_val.register(ops.StringJoin)
def _string_join(op, **kw):
    arg = map(partial(translate_val, **kw), op.arg)
    sep = translate_val(op.sep, **kw)
    elements = ", ".join(map(_sql, arg))
    return f"arrayStringConcat([{elements}], {sep})"


@translate_val.register(ops.StringConcat)
def _string_concat(op, **kw):
    arg = map(partial(translate_val, **kw), op.arg)
    args_formatted = ", ".join(map(_sql, arg))
    return f"arrayStringConcat([{args_formatted}])"


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


@translate_val.register(ops.Capitalize)
def _string_capitalize(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"CONCAT(UPPER(SUBSTR({arg}, 1, 1)), LOWER(SUBSTR({arg}, 2)))"


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
    expr = f"arrayStringConcat({call}, {sep})"
    return f"CASE WHEN empty({call}) THEN NULL ELSE {expr} END"


@translate_val.register(ops.StrRight)
def _string_right(op, **kw):
    arg = translate_val(op.arg, **kw)
    nchars = translate_val(op.nchars, **kw)
    nchars = _parenthesize(op.nchars, nchars)
    return f"substring({arg}, -{nchars})"


@translate_val.register(ops.Cot)
def _cotangent(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"1.0 / tan({arg})"


def _bit_agg(func):
    def _translate(op, **kw):
        arg = translate_val(op.arg, **kw)
        if not isinstance((type := op.arg.output_dtype), dt.UnsignedInteger):
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


@translate_val.register(ops.StructColumn)
def _struct_column(op, **kw):
    # ClickHouse struct types cannot be nullable
    # (non-nested fields can be nullable)
    values = translate_val(op.values, **kw)
    struct_type = serialize(op.output_dtype.copy(nullable=False))
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
    arg_dtype = arg.output_dtype
    arg = translate_val(op.arg, render_aliases=render_aliases, **kw)
    idx = arg_dtype.names.index(op.field)
    typ = arg_dtype.types[idx]
    return f"CAST({arg}.{idx + 1} AS {serialize(typ)})"


@translate_val.register(ops.NthValue)
def _nth_value(op, **kw):
    arg = translate_val(op.arg, **kw)
    nth = translate_val(op.nth, **kw)
    return f"nth_value({arg}, ({nth}) + 1)"


@translate_val.register(ops.Repeat)
def _repeat(op, **kw):
    arg = translate_val(op.arg, **kw)
    times = translate_val(op.times, **kw)
    return f"repeat({arg}, accurateCast({times}, 'UInt64'))"


@translate_val.register(ops.NullIfZero)
def _null_if_zero(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf({arg}, 0)"


@translate_val.register(ops.ZeroIfNull)
def _zero_if_null(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"ifNull({arg}, 0)"


@translate_val.register(ops.DayOfWeekIndex)
def _day_of_week_index(op, **kw):
    arg = translate_val(op.arg, **kw)
    weekdays = 7
    offset = f"toDayOfWeek({arg})"
    return f"((({offset} - 1) % {weekdays:d}) + {weekdays:d}) % {weekdays:d}"


@translate_val.register(ops.FloorDivide)
def _floor_divide(op, **kw):
    new_op = ops.Floor(ops.Divide(op.left, op.right))
    return translate_val(new_op, **kw)


@translate_val.register(ops.ScalarParameter)
def _scalar_param(op, params: Mapping[ops.Node, Any], **kw):
    raw_value = params[op]
    dtype = op.output_dtype
    if isinstance(dtype, dt.Struct):
        literal = ibis.struct(raw_value, type=dtype)
    elif isinstance(dtype, dt.Map):
        literal = ibis.map(raw_value, type=dtype)
    else:
        literal = ibis.literal(raw_value, type=dtype)
    return translate_val(literal.op(), **kw)


@translate_val.register(ops.StringContains)
def _string_contains(op, **kw):
    haystack = translate_val(op.haystack, **kw)
    needle = translate_val(op.needle, **kw)
    return f"locate({haystack}, {needle}) > 0"


def contains(op_string: Literal["IN", "NOT IN"]) -> str:
    def tr(op, *, cache, **kw):
        from ibis.backends.clickhouse.compiler import translate

        value = op.value
        options = op.options
        if isinstance(options, tuple) and not options:
            return {"NOT IN": "TRUE", "IN": "FALSE"}[op_string]

        left_arg = translate_val(value, **kw)
        if helpers.needs_parens(value):
            left_arg = helpers.parenthesize(left_arg)

        # special case non-foreign isin/notin expressions
        if not isinstance(options, tuple) and options.output_shape.is_columnar():
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


translate_val.register(ops.Contains)(contains("IN"))
translate_val.register(ops.NotContains)(contains("NOT IN"))


@translate_val.register(ops.DayOfWeekName)
def day_of_week_name(op, **kw):
    # ClickHouse 20 doesn't support dateName
    #
    # ClickHouse 21 supports dateName is broken for regexen:
    # https://github.com/ClickHouse/ClickHouse/issues/32777
    #
    # ClickHouses 20 and 21 also have a broken case statement hence the ifnull:
    # https://github.com/ClickHouse/ClickHouse/issues/32849
    #
    # We test against 20 in CI, so we implement day_of_week_name as follows
    arg = op.arg
    nullable = arg.output_dtype.nullable
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


@translate_val.register(ops.Greatest)
@translate_val.register(ops.Least)
@translate_val.register(ops.Coalesce)
def _vararg_func(op, **kw):
    args = ", ".join(map(_sql, map(partial(translate_val, **kw), op.arg)))
    return f"{op.__class__.__name__.lower()}({args})"


@translate_val.register(ops.Map)
def _map(op, **kw):
    keys = translate_val(op.keys, **kw)
    values = translate_val(op.values, **kw)
    typ = serialize(op.output_dtype)
    return f"CAST(({keys}, {values}) AS {typ})"


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

        if helpers.needs_parens(op_left):
            left = helpers.parenthesize(left)

        if helpers.needs_parens(op_right):
            right = helpers.parenthesize(right)

        return f"{left} {symbol} {right}"

    return formatter


_binary_infix_ops = {
    # Binary operations
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.Modulus: "%",
    # Comparisons
    ops.Equals: "=",
    ops.NotEquals: "!=",
    ops.GreaterEqual: ">=",
    ops.Greater: ">",
    ops.LessEqual: "<=",
    ops.Less: "<",
    # Boolean comparisons
    ops.And: "AND",
    ops.Or: "OR",
    ops.DateAdd: "+",
    ops.DateSub: "-",
    ops.DateDiff: "-",
    ops.TimestampAdd: "+",
    ops.TimestampSub: "-",
    ops.TimestampDiff: "-",
}


for _op, _sym in _binary_infix_ops.items():
    translate_val.register(_op)(_binary_infix(_sym))

del _op, _sym

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
    ops.Where: "if",
    ops.ArrayLength: "length",
    ops.ArrayConcat: "arrayConcat",
    ops.Unnest: "arrayJoin",
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.Strftime: "formatDateTime",
    ops.IsNull: "isNull",
    ops.NotNull: "isNotNull",
    ops.IfNull: "ifNull",
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
    ops.ArrayDistinct: "arrayDistinct",
    ops.ArraySort: "arraySort",
    ops.ArrayContains: "has",
    ops.FirstValue: "first_value",
    ops.LastValue: "last_value",
    ops.NTile: "ntile",
}


for _op, _name in _simple_ops.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):
        translate_val.register(_op)(_agg(_name))
    else:

        @translate_val.register(_op)
        def _fmt(op, _name: str = _name, **kw):
            args = ", ".join(map(_sql, map(partial(translate_val, **kw), op.args)))
            return f"{_name}({args})"


del _fmt, _name, _op


@translate_val.register(ops.ExtractMicrosecond)
def _extract_microsecond(op, **kw):
    arg = translate_val(op.arg, **kw)
    dtype = serialize(op.output_dtype)

    datetime_type_args = ["6"]
    if (tz := op.arg.output_dtype.timezone) is not None:
        datetime_type_args.append(f"'{tz}'")

    datetime_type = f"DateTime64({', '.join(datetime_type_args)})"
    return f"CAST(toUnixTimestamp64Micro(CAST({arg} AS {datetime_type})) % 1000000 AS {dtype})"


@translate_val.register(ops.ExtractMillisecond)
def _extract_millisecond(op, **kw):
    arg = translate_val(op.arg, **kw)
    dtype = serialize(op.output_dtype)

    datetime_type_args = ["3"]
    if (tz := op.arg.output_dtype.timezone) is not None:
        datetime_type_args.append(f"'{tz}'")

    datetime_type = f"DateTime64({', '.join(datetime_type_args)})"
    return f"CAST(toUnixTimestamp64Milli(CAST({arg} AS {datetime_type})) % 1000 AS {dtype})"


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
        return f'{value} PRECEDING'
    else:
        return f'{value} FOLLOWING'


def format_window_frame(func, frame, **kw):
    components = []

    if frame.how == "rows" and frame.max_lookback is not None:
        raise NotImplementedError(
            'Rows with max lookback is not implemented for the ClickHouse backend.'
        )

    if frame.group_by:
        partition_args = ', '.join(
            map(_sql, map(partial(translate_val, **kw), frame.group_by))
        )
        components.append(f'PARTITION BY {partition_args}')

    if frame.order_by:
        order_args = ', '.join(
            map(_sql, map(partial(translate_val, **kw), frame.order_by))
        )
        components.append(f'ORDER BY {order_args}')

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
            start = 'UNBOUNDED PRECEDING'
        else:
            start = format_window_boundary(frame.start, **kw)

        if frame.end is None:
            end = 'UNBOUNDED FOLLOWING'
        else:
            end = format_window_boundary(frame.end, **kw)

        frame = f'{frame.how.upper()} BETWEEN {start} AND {end}'
        components.append(frame)

    return f"OVER ({' '.join(components)})"


_map_interval_to_microseconds = {
    'W': 604800000000,
    'D': 86400000000,
    'h': 3600000000,
    'm': 60000000,
    's': 1000000,
    'ms': 1000,
    'us': 1,
    'ns': 0.001,
}

_map_interval_op_to_op = {
    # Literal Intervals have two args, i.e.
    # Literal(1, Interval(value_type=int8, unit='D', nullable=True))
    # Parse both args and multiply 1 * _map_interval_to_microseconds['D']
    ops.Literal: mul,
    ops.IntervalMultiply: mul,
    ops.IntervalAdd: add,
    ops.IntervalSubtract: sub,
}


UNSUPPORTED_REDUCTIONS = (
    ops.ApproxMedian,
    ops.GroupConcat,
    ops.ApproxCountDistinct,
)


@translate_val.register(ops.WindowFunction)
def _window(op: ops.WindowFunction, **kw: Any):
    if isinstance(op.func, UNSUPPORTED_REDUCTIONS):
        raise com.UnsupportedOperationError(
            f'{type(op.func)} is not supported in window functions'
        )

    if isinstance(op.func, ops.CumulativeOp):
        arg = cumulative_to_window(op.func, op.frame)
        return translate_val(arg, **kw)

    window_formatted = format_window_frame(op, op.frame, **kw)
    func = op.func.__window_op__
    func_formatted = translate_val(func, **kw)
    result = f'{func_formatted} {window_formatted}'

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
                offset_fmt = '1'
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


shift_like(ops.Lag, "lagInFrame")
shift_like(ops.Lead, "leadInFrame")


@translate_val.register(ops.RowNumber)
def _row_number(_, **kw):
    return "row_number()"


@translate_val.register(ops.DenseRank)
def _dense_rank(_, **kw):
    return "dense_rank()"


@translate_val.register(ops.MinRank)
def _rank(_, **kw):
    return "rank()"


@translate_val.register(ops.ExtractProtocol)
def _extract_protocol(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(protocol({arg}), '')"


@translate_val.register(ops.ExtractAuthority)
def _extract_authority(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(netloc({arg}), '')"


@translate_val.register(ops.ExtractHost)
def _extract_host(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(domain({arg}), '')"


@translate_val.register(ops.ExtractFile)
def _extract_file(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(cutFragment(pathFull({arg})), '')"


@translate_val.register(ops.ExtractPath)
def _extract_path(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(path({arg}), '')"


@translate_val.register(ops.ExtractQuery)
def _extract_query(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    if (key := op.key) is not None:
        key = translate_val(key, render_aliases=False, **kw)
        return f"nullIf(extractURLParameter({arg}, {key}), '')"
    else:
        return f"nullIf(queryString({arg}), '')"


@translate_val.register(ops.ExtractFragment)
def _extract_fragment(op, **kw):
    arg = translate_val(op.arg, render_aliases=False, **kw)
    return f"nullIf(fragment({arg}), '')"


@translate_val.register(ops.ArrayStringJoin)
def _array_string_join(op, **kw):
    arg = translate_val(op.arg, **kw)
    sep = translate_val(op.sep, **kw)
    return f"arrayStringConcat({arg}, {sep})"


@translate_val.register(ops.Argument)
def _argument(op, **_):
    return op.name


@translate_val.register(ops.ArrayMap)
def _array_map(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    return f"arrayMap(({op.parameter}) -> {result}, {arg})"


@translate_val.register(ops.ArrayFilter)
def _array_filter(op, **kw):
    arg = translate_val(op.arg, **kw)
    result = translate_val(op.result, **kw)
    return f"arrayFilter(({op.parameter}) -> {result}, {arg})"


@translate_val.register(ops.ArrayPosition)
def _array_position(op, **kw):
    arg = translate_val(op.arg, **kw)
    el = translate_val(op.other, **kw)
    return f"indexOf({arg}, {el}) - 1"


@translate_val.register(ops.ArrayRemove)
def _array_remove(op, **kw):
    return translate_val(ops.ArrayFilter(op.arg, flip(ops.NotEquals, op.other)), **kw)


@translate_val.register(ops.ArrayUnion)
def _array_union(op, **kw):
    return translate_val(ops.ArrayDistinct(ops.ArrayConcat(op.left, op.right)), **kw)


@translate_val.register(ops.ArrayZip)
def _array_zip(op: ops.ArrayZip, **kw: Any) -> str:
    arglist = []
    for arg in op.arg:
        sql_arg = translate_val(arg, **kw)
        with contextlib.suppress(AttributeError):
            sql_arg = sql_arg.sql(dialect="clickhouse")
        arglist.append(sql_arg)
    return f"arrayZip({', '.join(arglist)})"
