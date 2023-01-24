from __future__ import annotations

import calendar
import functools
from functools import partial
from operator import add, mul, sub
from typing import Any, Literal, Mapping

import sqlglot as sg

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.backends.base.sql.registry import helpers
from ibis.backends.clickhouse.datatypes import serialize

# TODO: Ideally we can translate bottom up a la `relations.py`
# TODO: Find a way to remove all the dialect="clickhouse" kwargs


@functools.singledispatch
def translate_val(op, **_):
    """Translate a value expression into sqlglot."""
    raise NotImplementedError(type(op))


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
def _alias(op, **kw):
    val = translate_val(op.arg, **kw)
    return sg.alias(val, op.name, dialect="clickhouse")


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
        suffix = _interval_cast_suffixes[op.to.unit]
        return f"toInterval{suffix}({arg})"

    to = translate_val(op.to, **kw)
    return f"CAST({arg} AS {to})"


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
    return translate_val(ops.Not(ops.Any(op.arg)), **kw)


@translate_val.register(ops.NotAll)
def _not_all(op, **kw):
    return translate_val(ops.Not(ops.All(op.arg)), **kw)


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
    if (length := op.length) is None:
        return f"substring({arg}, {start} + 1)"

    length = translate_val(length, **kw)
    return f"substring({arg}, {start} + 1, {length})"


@translate_val.register(ops.StringFind)
def _string_find(op, **kw):
    if op.start is not None:
        raise com.UnsupportedOperationError(
            "String find doesn't support start argument"
        )
    if op.end is not None:
        raise com.UnsupportedOperationError("String find doesn't support end argument")

    arg = translate_val(op.arg, **kw)
    substr = translate_val(op.substr, **kw)
    return f"locate({arg}, {substr}) - 1"


@translate_val.register(ops.RegexExtract)
def _regex_extract(op, **kw):
    arg = translate_val(op.arg, **kw)
    pattern = translate_val(op.pattern, **kw)
    index = "Null" if op.index is None else translate_val(op.index, **kw)

    # arg can be Nullable, which is not allowed in extractAll, so cast to non
    # nullable type
    arg = f"CAST({arg} AS String)"

    # extract all matches in pattern
    extracted = f"CAST(extractAll({arg}, {pattern}) AS Array(Nullable(String)))"

    # if there's a match
    #   if the index IS zero or null
    #     return the full string
    #   else
    #     return the Nth match group
    # else
    #   return null
    does_match = f"match({arg}, {pattern})"
    idx = f"CAST(nullIf({index}, 0) AS Nullable(Int64))"
    then = f"if({idx} IS NULL, {arg}, {extracted}[{idx}])"
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
    if dtype.unit in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    return f"INTERVAL {op.value} {dtype.resolution.upper()}"


@translate_val.register(ops.IntervalFromInteger)
def _interval_from_integer(op, **kw):
    dtype = op.output_dtype
    if dtype.unit in {"ms", "us", "ns"}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    arg = translate_val(op.arg, **kw)
    return f"INTERVAL {arg} {dtype.resolution.upper()}"


@translate_val.register(ops.Literal)
def _literal(op, **kw):
    value = op.value
    dtype = op.output_dtype
    if value is None and op.output_dtype.nullable:
        return _null_literal(op)
    if dtype.is_boolean():
        return str(int(bool(value)))
    elif dtype.is_inet():
        v = str(value)
        return f"toIPv6({v!r})" if ":" in v else f"toIPv4({v!r})"
    elif dtype.is_string():
        quoted = value.replace("'", "''").replace("\\", "\\\\")
        return f"'{quoted}'"
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

        if (timezone := op.output_dtype.timezone) is not None:
            args.append(timezone)

        joined_args = ", ".join(map(repr, args))
        return f"{func}({joined_args})"

    elif isinstance(op.output_dtype, dt.Date):
        formatted = value.strftime('%Y-%m-%d')
        return f"toDate('{formatted}')"
    elif isinstance(op.output_dtype, dt.Array):
        values = ", ".join(_array_literal_values(op))
        return f"[{values}]"
    elif isinstance(op.output_dtype, dt.Map):
        values = ", ".join(_map_literal_values(op))
        return f"map({values})"
    elif isinstance(op.output_dtype, dt.Set):
        args = ", ".join(map(repr, value))
        return f"({args})"
    elif isinstance(op.output_dtype, dt.Struct):
        fields = ", ".join(f"{value} as `{key}`" for key, value in op.value.items())
        return f"tuple({fields})"
    else:
        raise NotImplementedError(type(op))


def _array_literal_values(op):
    value_type = op.output_dtype.value_type
    for v in op.value:
        value = ops.Literal(v, dtype=value_type)
        yield _literal(value)


def _map_literal_values(op):
    value_type = op.output_dtype.value_type
    for k, v in op.value.items():
        value = ops.Literal(v, dtype=value_type)
        yield repr(k)
        yield _literal(value)


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
    if (unit := op.unit) in {"ms", "us", "ns"}:
        raise ValueError(f"`{unit}` unit is not supported!")

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

    unit = op.unit
    arg = translate_val(op.arg, **kw)
    try:
        converter = converters[unit]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

    return f"{converter}({arg})"


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
def _struct_field(op, **kw):
    arg = op.arg
    arg_dtype = arg.output_dtype
    arg = translate_val(op.arg, **kw)
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


@translate_val.register(ops.NullLiteral)
def _null_literal(_, **__):
    return "Null"


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
    def translate(op, *, cache, **kw):
        import ibis.expr.analysis as an

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
            leaves = list(an.find_immediate_parent_tables(options))
            nleaves = len(leaves)
            if nleaves > 1:
                raise NotImplementedError(
                    "more than one leaf table in a NOT IN/IN query unsupported"
                )
            (leaf,) = leaves

            shared_roots_count = sum(
                an.shares_all_roots(value, child)
                for child in an.find_immediate_parent_tables(options)
            )
            if shared_roots_count == nleaves:
                from ibis.backends.clickhouse.compiler.relations import translate_rel

                op = options.to_expr().as_table().op()
                subquery = translate_rel(op, table=cache[leaf], **kw)
                right_arg = f"({subquery})"
            else:
                raise NotImplementedError(
                    "ClickHouse doesn't support correlated subqueries"
                )
        else:
            right_arg = translate_val(options, cache=cache, **kw)

        # we explicitly do NOT parenthesize the right side because it doesn't
        # make sense to do so for Sequence operations
        return f"{left_arg} {op_string} {right_arg}"

    return translate


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
    # Unary aggregates
    ops.ApproxMedian: "median",
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
    ops.RegexSearch: "match",
    ops.RegexReplace: "replaceRegexpAll",
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
    ops.MapKeys: "mapKeys",
    ops.MapValues: "mapValues",
    ops.MapMerge: "mapUpdate",
    ops.BitwiseAnd: "bitAnd",
    ops.BitwiseOr: "bitOr",
    ops.BitwiseXor: "bitXor",
    ops.BitwiseLeftShift: "bitShiftLeft",
    ops.BitwiseRightShift: "bitShiftRight",
    ops.BitwiseNot: "bitNot",
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


def cumulative_to_window(op, window):
    klass = _cumulative_to_reduction[type(op)]
    new_op = klass(*op.args)
    win = ibis.cumulative_window().group_by(window._group_by).order_by(window._order_by)
    new_expr = an.windowize_function(new_op.to_expr(), win)
    return new_expr.op()


def format_window(op, window, **kw):
    components = []

    if window.max_lookback is not None:
        raise NotImplementedError(
            'Rows with max lookback is not implemented for string-based backends.'
        )

    if window._group_by:
        partition_args = ', '.join(
            map(_sql, map(partial(translate_val, **kw), window._group_by))
        )
        components.append(f'PARTITION BY {partition_args}')

    if window._order_by:
        order_args = ', '.join(
            map(_sql, map(partial(translate_val, **kw), window._order_by))
        )
        components.append(f'ORDER BY {order_args}')

    p, f = window.preceding, window.following

    def _prec(p: int | None) -> str:
        assert p is None or p >= 0

        if p is None:
            prefix = 'UNBOUNDED'
        else:
            if not p:
                return 'CURRENT ROW'
            prefix = str(p)
        return f'{prefix} PRECEDING'

    def _foll(f: int | None) -> str:
        assert f is None or f >= 0

        if f is None:
            prefix = 'UNBOUNDED'
        else:
            if not f:
                return 'CURRENT ROW'
            prefix = str(f)

        return f'{prefix} FOLLOWING'

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

    if isinstance(op.expr, frame_clause_not_allowed):
        frame = None
    elif p is not None and f is not None:
        start = _prec(p)
        end = _foll(f)
        frame = f'{window.how.upper()} BETWEEN {start} AND {end}'

    elif p is not None:
        if isinstance(p, tuple):
            start, end = map(_prec, p)
            frame = f'{window.how.upper()} BETWEEN {start} AND {end}'
        else:
            kind = 'ROWS' if p > 0 else 'RANGE'
            frame = f'{kind} BETWEEN {_prec(p)} AND UNBOUNDED FOLLOWING'
    elif f is not None:
        if isinstance(f, tuple):
            start, end = map(_foll, f)
            frame = f'{window.how.upper()} BETWEEN {start} AND {end}'
        else:
            kind = 'ROWS' if f > 0 else 'RANGE'
            frame = f'{kind} BETWEEN UNBOUNDED PRECEDING AND {_foll(f)}'
    else:
        # no-op, default is full sample
        frame = None

    if frame is not None:
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
    # Parse both args and multipy 1 * _map_interval_to_microseconds['D']
    ops.Literal: mul,
    ops.IntervalMultiply: mul,
    ops.IntervalAdd: add,
    ops.IntervalSubtract: sub,
}


def _replace_interval_with_scalar(expr: ir.Expr | dt.Interval | float):
    if isinstance(expr, ir.Expr):
        expr_op = expr.op()
    else:
        expr_op = None

    if not isinstance(expr, (dt.Interval, ir.IntervalValue)):
        # Literal expressions have op method but native types do not.
        if isinstance(expr_op, ops.Literal):
            return expr_op.value
        else:
            return expr
    elif isinstance(expr, dt.Interval):
        try:
            microseconds = _map_interval_to_microseconds[expr.unit]
            return microseconds
        except KeyError:
            raise ValueError(
                "Expected preceding values of week(), "
                "day(), hour(), minute(), second(), millisecond(), "
                f"microseconds(), nanoseconds(); got {expr}"
            )
    elif expr_op.args and isinstance(expr, ir.IntervalValue):
        if len(expr_op.args) > 2:
            raise NotImplementedError("'preceding' argument cannot be parsed.")
        left_arg = _replace_interval_with_scalar(expr_op.args[0])
        right_arg = _replace_interval_with_scalar(expr_op.args[1])
        method = _map_interval_op_to_op[type(expr_op)]
        return method(left_arg, right_arg)
    else:
        raise TypeError(f'expr has unknown type {type(expr).__name__}')


def time_range_to_range_window(window):
    # Check that ORDER BY column is a single time column:
    order_by_vars = [x.op().args[0] for x in window._order_by]
    if len(order_by_vars) > 1:
        raise com.IbisInputError(
            f"Expected 1 order-by variable, got {len(order_by_vars)}"
        )

    order_var = window._order_by[0].op().args[0]
    timestamp_order_var = order_var.cast('int64')
    window = window._replace(order_by=timestamp_order_var, how='range')

    # Need to change preceding interval expression to scalars
    preceding = window.preceding
    if isinstance(preceding, ir.IntervalScalar):
        new_preceding = _replace_interval_with_scalar(preceding)
        window = window._replace(preceding=new_preceding)

    return window


@functools.singledispatch
def transform_result(_, expr) -> str:
    return expr


@transform_result.register(ops.RowNumber)
@transform_result.register(ops.DenseRank)
@transform_result.register(ops.MinRank)
@transform_result.register(ops.NTile)
def _(_, expr) -> str:
    return f"({expr} - 1)"


REQUIRE_ORDER_BY = (
    ops.DenseRank,
    ops.MinRank,
    ops.FirstValue,
    ops.LastValue,
    ops.PercentRank,
    ops.CumeDist,
    ops.NTile,
)

UNSUPPORTED_REDUCTIONS = (
    ops.ApproxMedian,
    ops.GroupConcat,
    ops.ApproxCountDistinct,
)


@translate_val.register(ops.Window)
def _window(op: ops.Window, **kw: Any):
    arg = op.expr
    window = op.window

    if isinstance(arg, UNSUPPORTED_REDUCTIONS):
        raise com.UnsupportedOperationError(
            f'{type(arg)} is not supported in window functions'
        )

    if isinstance(arg, ops.CumulativeOp):
        arg = cumulative_to_window(arg, window)
        return translate_val(arg, **kw)

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if isinstance(arg, REQUIRE_ORDER_BY) and not window._order_by:
        window = window.order_by(arg.args[0])

    # Time ranges need to be converted to microseconds.
    # FIXME(kszucs): avoid the expression roundtrip
    if window.how == 'range':
        order_by_types = [type(x.op().args[0]) for x in window._order_by]
        time_range_types = (ir.TimeColumn, ir.DateColumn, ir.TimestampColumn)
        if any(col_type in time_range_types for col_type in order_by_types):
            window = time_range_to_range_window(window)

    window_formatted = format_window(op, window, **kw)

    arg_formatted = translate_val(arg, **kw)
    result = f'{arg_formatted} {window_formatted}'

    return transform_result(arg, result)


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


@translate_val.register(ops.NTile)
def _ntile(op, **kw):
    return f'ntile({translate_val(op.buckets, **kw)})'


@translate_val.register(ops.RowNumber)
def _row_number(_, **kw):
    return "row_number()"


@translate_val.register(ops.DenseRank)
def _dense_rank(_, **kw):
    return "dense_rank()"


@translate_val.register(ops.MinRank)
def _rank(_, **kw):
    return "rank()"


@translate_val.register(ops.FirstValue)
def _first_value(op, **kw):
    return f"first_value({translate_val(op.arg, **kw)})"


@translate_val.register(ops.LastValue)
def _last_value(op, **kw):
    return f"last_value({translate_val(op.arg, **kw)})"


@translate_val.register(ops.ExtractProtocol)
def _extract_protocol(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(protocol({arg}), '')"


@translate_val.register(ops.ExtractAuthority)
def _extract_authority(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(netloc({arg}), '')"


@translate_val.register(ops.ExtractHost)
def _extract_host(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(domain({arg}), '')"


@translate_val.register(ops.ExtractFile)
def _extract_file(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(cutFragment(pathFull({arg})), '')"


@translate_val.register(ops.ExtractPath)
def _extract_path(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(path({arg}), '')"


@translate_val.register(ops.ExtractQuery)
def _extract_query(op, **kw):
    arg = translate_val(op.arg, **kw)
    if (key := op.key) is not None:
        key = translate_val(key, **kw)
        return f"nullIf(extractURLParameter({arg}, {key}), '')"
    else:
        return f"nullIf(queryString({arg}), '')"


@translate_val.register(ops.ExtractFragment)
def _extract_fragment(op, **kw):
    arg = translate_val(op.arg, **kw)
    return f"nullIf(fragment({arg}), '')"
