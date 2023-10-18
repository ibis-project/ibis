from __future__ import annotations

import functools
import itertools
import locale
import operator
import platform
import re
import string

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import ibis.backends.base.sql.registry.geospatial as geo
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

# used for literal translate
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    get_sqla_table,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
    varargs,
)
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.registry import (
    _bitwise_op,
    _extract,
    geospatial_functions,
    get_col,
)

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)

if geospatial_supported:
    operation_registry.update(geospatial_functions)


_truncate_precisions = {
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "second",
    "m": "minute",
    "h": "hour",
    "D": "day",
    "W": "week",
    "M": "month",
    "Q": "quarter",
    "Y": "year",
}


def _timestamp_truncate(t, op):
    sa_arg = t.translate(op.arg)
    try:
        precision = _truncate_precisions[op.unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {op.unit!r}")
    return sa.func.date_trunc(precision, sa_arg)


def _timestamp_bucket(t, op):
    arg = t.translate(op.arg)
    interval = t.translate(op.interval)

    origin = sa.literal_column("timestamp '1970-01-01 00:00:00'")

    if op.offset is not None:
        origin = origin + t.translate(op.offset)
    return sa.func.date_bin(interval, arg, origin)


def _typeof(t, op):
    sa_arg = t.translate(op.arg)
    typ = sa.cast(sa.func.pg_typeof(sa_arg), sa.TEXT)

    # select pg_typeof('thing') returns unknown so we have to check the child's
    # type for nullness
    return sa.case(
        ((typ == "unknown") & (op.arg.dtype != dt.null), "text"),
        ((typ == "unknown") & (op.arg.dtype == dt.null), "null"),
        else_=typ,
    )


_strftime_to_postgresql_rules = {
    "%a": "TMDy",  # TM does it in a locale dependent way
    "%A": "TMDay",
    "%w": "D",  # 1-based day of week, see below for how we make this 0-based
    "%d": "DD",  # day of month
    "%-d": "FMDD",  # - is no leading zero for Python same for FM in postgres
    "%b": "TMMon",  # Sep
    "%B": "TMMonth",  # September
    "%m": "MM",  # 01
    "%-m": "FMMM",  # 1
    "%y": "YY",  # 15
    "%Y": "YYYY",  # 2015
    "%H": "HH24",  # 09
    "%-H": "FMHH24",  # 9
    "%I": "HH12",  # 09
    "%-I": "FMHH12",  # 9
    "%p": "AM",  # AM or PM
    "%M": "MI",  # zero padded minute
    "%-M": "FMMI",  # Minute
    "%S": "SS",  # zero padded second
    "%-S": "FMSS",  # Second
    "%f": "US",  # zero padded microsecond
    "%z": "OF",  # utf offset
    "%Z": "TZ",  # uppercase timezone name
    "%j": "DDD",  # zero padded day of year
    "%-j": "FMDDD",  # day of year
    "%U": "WW",  # 1-based week of year
    # 'W': ?,  # meh
}

try:
    _strftime_to_postgresql_rules.update(
        {
            "%c": locale.nl_langinfo(locale.D_T_FMT),  # locale date and time
            "%x": locale.nl_langinfo(locale.D_FMT),  # locale date
            "%X": locale.nl_langinfo(locale.T_FMT),  # locale time
        }
    )
except AttributeError:
    HAS_LANGINFO = False
else:
    HAS_LANGINFO = True


# translate strftime spec into mostly equivalent PostgreSQL spec
_scanner = re.Scanner(  # type: ignore # re does have a Scanner attribute
    # double quotes need to be escaped
    [('"', lambda *_: r"\"")]
    + [
        (
            "|".join(
                map(
                    "(?:{})".format,
                    itertools.chain(
                        _strftime_to_postgresql_rules.keys(),
                        [
                            # "%e" is in the C standard and Python actually
                            # generates this if your spec contains "%c" but we
                            # don't officially support it as a specifier so we
                            # need to special case it in the scanner
                            "%e",
                            r"\s+",
                            rf"[{re.escape(string.punctuation)}]",
                            rf"[^{re.escape(string.punctuation)}\s]+",
                        ],
                    ),
                )
            ),
            lambda _, token: token,
        )
    ]
)


_lexicon_values = frozenset(_strftime_to_postgresql_rules.values())

_locale_specific_formats = frozenset(["%c", "%x", "%X"])
_strftime_blacklist = frozenset(["%w", "%U", "%e"]) | _locale_specific_formats


def _reduce_tokens(tokens, arg):
    # current list of tokens
    curtokens = []

    # reduced list of tokens that accounts for blacklisted values
    reduced = []

    non_special_tokens = frozenset(_strftime_to_postgresql_rules) - _strftime_blacklist

    # TODO: how much of a hack is this?
    for token in tokens:
        if token in _locale_specific_formats and not HAS_LANGINFO:
            raise com.UnsupportedOperationError(
                f"Format string component {token!r} is not supported on {platform.system()}"
            )
        # we are a non-special token %A, %d, etc.
        if token in non_special_tokens:
            curtokens.append(_strftime_to_postgresql_rules[token])

        # we have a string like DD, to escape this we
        # surround it with double quotes
        elif token in _lexicon_values:
            curtokens.append(f'"{token}"')

        # we have a token that needs special treatment
        elif token in _strftime_blacklist:
            if token == "%w":
                value = sa.extract("dow", arg)  # 0 based day of week
            elif token == "%U":
                value = sa.cast(sa.func.to_char(arg, "WW"), sa.SMALLINT) - 1
            elif token in ("%c", "%x", "%X"):
                # re scan and tokenize this pattern
                try:
                    new_pattern = _strftime_to_postgresql_rules[token]
                except KeyError:
                    raise ValueError(
                        "locale specific date formats (%%c, %%x, %%X) are "
                        "not yet implemented for %s" % platform.system()
                    )

                new_tokens, _ = _scanner.scan(new_pattern)
                value = functools.reduce(
                    sa.sql.ColumnElement.concat,
                    _reduce_tokens(new_tokens, arg),
                )
            elif token == "%e":
                # pad with spaces instead of zeros
                value = sa.func.replace(sa.func.to_char(arg, "DD"), "0", " ")

            reduced += [
                sa.func.to_char(arg, "".join(curtokens)),
                sa.cast(value, sa.TEXT),
            ]

            # empty current token list in case there are more tokens
            del curtokens[:]

        # uninteresting text
        else:
            curtokens.append(token)
    # append result to r if we had more tokens or if we have no
    # blacklisted tokens
    if curtokens:
        reduced.append(sa.func.to_char(arg, "".join(curtokens)))
    return reduced


def _strftime(arg, pattern):
    tokens, _ = _scanner.scan(pattern.value)
    reduced = _reduce_tokens(tokens, arg)
    return functools.reduce(sa.sql.ColumnElement.concat, reduced)


def _find_in_set(t, op):
    # TODO
    # this operation works with any type, not just strings. should the
    # operation itself also have this property?
    return (
        sa.func.coalesce(
            sa.func.array_position(
                pg.array(list(map(t.translate, op.values))),
                t.translate(op.needle),
            ),
            0,
        )
        - 1
    )


def _log(t, op):
    arg, base = op.args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        return sa.cast(
            sa.func.log(sa.cast(sa_base, sa.NUMERIC), sa.cast(sa_arg, sa.NUMERIC)),
            t.get_sqla_type(op.dtype),
        )
    return sa.func.ln(sa_arg)


def _regex_extract(arg, pattern, index):
    # wrap in parens to support 0th group being the whole string
    pattern = "(" + pattern + ")"
    # arrays are 1-based in postgres
    index = index + 1
    does_match = sa.func.textregexeq(arg, pattern)
    matches = sa.func.regexp_match(arg, pattern, type_=pg.ARRAY(sa.TEXT))
    return sa.case((does_match, matches[index]), else_=None)


def _array_repeat(t, op):
    """Repeat an array."""
    arg = t.translate(op.arg)
    times = t.translate(op.times)

    array_length = sa.func.cardinality(arg)
    array = sa.sql.elements.Grouping(arg) if isinstance(op.arg, ops.Literal) else arg

    # sequence from 1 to the total number of elements desired in steps of 1.
    series = sa.func.generate_series(1, times * array_length).table_valued()

    # if our current index modulo the array's length is a multiple of the
    # array's length, then the index is the array's length
    index = sa.func.coalesce(
        sa.func.nullif(series.column % array_length, 0), array_length
    )

    # tie it all together in a scalar subquery and collapse that into an ARRAY
    return sa.func.array(sa.select(array[index]).scalar_subquery())


def _table_column(t, op):
    ctx = t.context
    table = op.table

    sa_table = get_sqla_table(ctx, table)
    out_expr = get_col(sa_table, op)

    if op.dtype.is_timestamp():
        timezone = op.dtype.timezone
        if timezone is not None:
            out_expr = out_expr.op("AT TIME ZONE")(timezone).label(op.name)

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if t.permit_subquery and ctx.is_foreign_expr(table):
        return sa.select(out_expr)

    return out_expr


def _round(t, op):
    arg, digits = op.args
    sa_arg = t.translate(arg)

    if digits is None:
        return sa.func.round(sa_arg)

    # postgres doesn't allow rounding of double precision values to a specific
    # number of digits (though simple truncation on doubles is allowed) so
    # we cast to numeric and then cast back if necessary
    result = sa.func.round(sa.cast(sa_arg, sa.NUMERIC), t.translate(digits))
    if digits is not None and arg.dtype.is_decimal():
        return result
    result = sa.cast(result, pg.DOUBLE_PRECISION())
    return result


def _mod(t, op):
    left, right = map(t.translate, op.args)

    # postgres doesn't allow modulus of double precision values, so upcast and
    # then downcast later if necessary
    if not op.dtype.is_integer():
        left = sa.cast(left, sa.NUMERIC)
        right = sa.cast(right, sa.NUMERIC)

    result = left % right
    if op.dtype.is_float64():
        return sa.cast(result, pg.DOUBLE_PRECISION())
    else:
        return result


def _neg_idx_to_pos(array, idx):
    return sa.case((idx < 0, sa.func.cardinality(array) + idx), else_=idx)


def _array_slice(*, index_converter, array_length, func):
    def translate(t, op):
        arg = t.translate(op.arg)

        arg_length = array_length(arg)

        if (start := op.start) is None:
            start = 0
        else:
            start = t.translate(start)
            start = sa.func.least(arg_length, index_converter(arg, start))

        if (stop := op.stop) is None:
            stop = arg_length
        else:
            stop = index_converter(arg, t.translate(stop))

        return func(arg, start + 1, stop)

    return translate


def _array_index(*, index_converter, func):
    def translate(t, op):
        sa_array = t.translate(op.arg)
        sa_index = t.translate(op.index)
        if isinstance(op.arg, ops.Literal):
            sa_array = sa.sql.elements.Grouping(sa_array)
        return func(sa_array, index_converter(sa_array, sa_index) + 1)

    return translate


def _literal(t, op):
    dtype = op.dtype
    value = op.value

    if dtype.is_interval():
        return sa.literal_column(f"INTERVAL '{value} {dtype.resolution}'")
    elif dtype.is_geospatial():
        # inline_metadata ex: 'SRID=4326;POINT( ... )'
        return sa.literal_column(geo.translate_literal(op, inline_metadata=True))
    elif dtype.is_array():
        return pg.array(value)
    elif dtype.is_map():
        return pg.hstore(list(value.keys()), list(value.values()))
    else:
        return sa.literal(value)


def _string_agg(t, op):
    agg = sa.func.string_agg(t.translate(op.arg), t.translate(op.sep))
    if (where := op.where) is not None:
        return agg.filter(t.translate(where))
    return agg


def _corr(t, op):
    if op.how == "sample":
        raise ValueError(
            f"{t.__class__.__name__} only implements population correlation "
            "coefficient"
        )
    return _binary_variance_reduction(sa.func.corr)(t, op)


def _covar(t, op):
    suffix = {"sample": "samp", "pop": "pop"}
    how = suffix.get(op.how, "samp")
    func = getattr(sa.func, f"covar_{how}")
    return _binary_variance_reduction(func)(t, op)


def _mode(t, op):
    arg = op.arg
    if (where := op.where) is not None:
        arg = ops.IfElse(where, arg, None)
    return sa.func.mode().within_group(t.translate(arg))


def _quantile(t, op):
    arg = op.arg
    if (where := op.where) is not None:
        arg = ops.IfElse(where, arg, None)
    return sa.func.percentile_cont(t.translate(op.quantile)).within_group(
        t.translate(arg)
    )


def _median(t, op):
    arg = op.arg
    if (where := op.where) is not None:
        arg = ops.IfElse(where, arg, None)

    return sa.func.percentile_cont(0.5).within_group(t.translate(arg))


def _binary_variance_reduction(func):
    def variance_compiler(t, op):
        x = op.left
        if (x_type := x.dtype).is_boolean():
            x = ops.Cast(x, dt.Int32(nullable=x_type.nullable))

        y = op.right
        if (y_type := y.dtype).is_boolean():
            y = ops.Cast(y, dt.Int32(nullable=y_type.nullable))

        if t._has_reduction_filter_syntax:
            result = func(t.translate(x), t.translate(y))

            if (where := op.where) is not None:
                return result.filter(t.translate(where))
            return result
        else:
            if (where := op.where) is not None:
                x = ops.IfElse(where, x, None)
                y = ops.IfElse(where, y, None)
            return func(t.translate(x), t.translate(y))

    return variance_compiler


def _arg_min_max(sort_func):
    def translate(t, op: ops.ArgMin | ops.ArgMax) -> str:
        arg = t.translate(op.arg)
        key = t.translate(op.key)

        conditions = [arg != sa.null(), key != sa.null()]

        agg = sa.func.array_agg(pg.aggregate_order_by(arg, sort_func(key)))

        if (where := op.where) is not None:
            conditions.append(t.translate(where))
        return agg.filter(sa.and_(*conditions))[1]

    return translate


def _arbitrary(t, op):
    if (how := op.how) == "heavy":
        raise com.UnsupportedOperationError(
            f"postgres backend doesn't support how={how!r} for the arbitrary() aggregate"
        )
    func = getattr(sa.func, op.how)
    return t._reduction(func, op)


class struct_field(GenericFunction):
    inherit_cache = True


@compiles(struct_field)
def compile_struct_field_postgresql(element, compiler, **kw):
    arg, field = element.clauses
    return f"({compiler.process(arg, **kw)}).{field.name}"


def _struct_field(t, op):
    arg = op.arg
    idx = arg.dtype.names.index(op.field) + 1
    field_name = sa.literal_column(f"f{idx:d}")
    return struct_field(t.translate(arg), field_name, type_=t.get_sqla_type(op.dtype))


def _struct_column(t, op):
    types = op.dtype.types
    return sa.func.row(
        # we have to cast here, otherwise postgres refuses to allow the statement
        *map(t.translate, map(ops.Cast, op.values, types)),
        type_=t.get_sqla_type(
            dt.Struct({f"f{i:d}": typ for i, typ in enumerate(types, start=1)})
        ),
    )


def _unnest(t, op):
    arg = op.arg
    row_type = arg.dtype.value_type

    types = getattr(row_type, "types", (row_type,))

    is_struct = row_type.is_struct()
    derived = (
        sa.func.unnest(t.translate(arg))
        .table_valued(
            *(
                sa.column(f"f{i:d}", stype)
                for i, stype in enumerate(map(t.get_sqla_type, types), start=1)
            )
        )
        .render_derived(with_types=is_struct)
    )

    # wrap in a row column so that we can return a single column from this rule
    if not is_struct:
        return derived.c[0]
    return sa.func.row(*derived.c)


def _array_sort(arg):
    flat = sa.func.unnest(arg).column_valued()
    return sa.func.array(sa.select(flat).order_by(flat).scalar_subquery())


def _array_position(haystack, needle):
    t = (
        sa.func.unnest(haystack)
        .table_valued("value", with_ordinality="idx", name="haystack")
        .render_derived()
    )
    idx = t.c.idx - 1
    return sa.func.coalesce(
        sa.select(idx).where(t.c.value == needle).limit(1).scalar_subquery(), -1
    )


def _array_map(t, op):
    return sa.func.array(
        # this translates to the function call, with column names the same as
        # the parameter names in the lambda
        sa.select(t.translate(op.body))
        .select_from(
            # unnest the input array
            sa.func.unnest(t.translate(op.arg))
            # name the columns of the result the same as the lambda parameter
            # so that we can reference them as such in the outer query
            .table_valued(op.param).render_derived()
        )
        .scalar_subquery()
    )


def _array_filter(t, op):
    param = op.param
    return sa.func.array(
        sa.select(sa.column(param, type_=t.get_sqla_type(op.arg.dtype.value_type)))
        .select_from(
            sa.func.unnest(t.translate(op.arg)).table_valued(param).render_derived()
        )
        .where(t.translate(op.body))
        .scalar_subquery()
    )


operation_registry.update(
    {
        ops.Literal: _literal,
        # We override this here to support time zones
        ops.TableColumn: _table_column,
        ops.Argument: lambda t, op: sa.column(op.name, type_=t.get_sqla_type(op.dtype)),
        # types
        ops.TypeOf: _typeof,
        # Floating
        ops.IsNan: fixed_arity(lambda arg: arg == float("nan"), 1),
        ops.IsInf: fixed_arity(
            lambda arg: sa.or_(arg == float("inf"), arg == float("-inf")), 1
        ),
        # boolean reductions
        ops.Any: reduction(sa.func.bool_or),
        ops.All: reduction(sa.func.bool_and),
        # strings
        ops.GroupConcat: _string_agg,
        ops.Capitalize: unary(sa.func.initcap),
        ops.RegexSearch: fixed_arity(lambda x, y: x.op("~")(y), 2),
        # postgres defaults to replacing only the first occurrence
        ops.RegexReplace: fixed_arity(
            lambda string, pattern, replacement: sa.func.regexp_replace(
                string, pattern, replacement, "g"
            ),
            3,
        ),
        ops.Translate: fixed_arity(sa.func.translate, 3),
        ops.RegexExtract: fixed_arity(_regex_extract, 3),
        ops.StringSplit: fixed_arity(
            lambda col, sep: sa.func.string_to_array(
                col, sep, type_=sa.ARRAY(col.type)
            ),
            2,
        ),
        ops.FindInSet: _find_in_set,
        # math
        ops.Log: _log,
        ops.Log2: unary(lambda x: sa.func.log(2, x)),
        ops.Log10: unary(sa.func.log),
        ops.Round: _round,
        ops.Modulus: _mod,
        # dates and times
        ops.DateFromYMD: fixed_arity(sa.func.make_date, 3),
        ops.DateTruncate: _timestamp_truncate,
        ops.TimestampTruncate: _timestamp_truncate,
        ops.TimestampBucket: _timestamp_bucket,
        ops.IntervalFromInteger: (
            lambda t, op: t.translate(op.arg)
            * sa.text(f"INTERVAL '1 {op.dtype.resolution}'")
        ),
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(operator.sub, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: fixed_arity(operator.sub, 2),
        ops.Strftime: fixed_arity(_strftime, 2),
        ops.ExtractEpochSeconds: fixed_arity(
            lambda arg: sa.cast(sa.extract("epoch", arg), sa.INTEGER), 1
        ),
        ops.ExtractDayOfYear: _extract("doy"),
        ops.ExtractWeekOfYear: _extract("week"),
        # extracting the second gives us the fractional part as well, so smash that
        # with a cast to SMALLINT
        ops.ExtractSecond: fixed_arity(
            lambda arg: sa.cast(sa.func.floor(sa.extract("second", arg)), sa.SMALLINT),
            1,
        ),
        # we get total number of milliseconds including seconds with extract so we
        # mod 1000
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.cast(
                sa.func.floor(sa.extract("millisecond", arg)) % 1000,
                sa.SMALLINT,
            ),
            1,
        ),
        ops.DayOfWeekIndex: fixed_arity(
            lambda arg: sa.cast(
                sa.cast(sa.extract("dow", arg) + 6, sa.SMALLINT) % 7, sa.SMALLINT
            ),
            1,
        ),
        ops.DayOfWeekName: fixed_arity(
            lambda arg: sa.func.trim(sa.func.to_char(arg, "Day")), 1
        ),
        ops.TimeFromHMS: fixed_arity(sa.func.make_time, 3),
        # array operations
        ops.ArrayLength: unary(sa.func.cardinality),
        ops.ArrayCollect: reduction(sa.func.array_agg),
        ops.ArrayColumn: (lambda t, op: pg.array(list(map(t.translate, op.cols)))),
        ops.ArraySlice: _array_slice(
            index_converter=_neg_idx_to_pos,
            array_length=sa.func.cardinality,
            func=lambda arg, start, stop: arg[start:stop],
        ),
        ops.ArrayIndex: _array_index(
            index_converter=_neg_idx_to_pos, func=lambda arg, index: arg[index]
        ),
        ops.ArrayConcat: varargs(lambda *args: functools.reduce(operator.add, args)),
        ops.ArrayRepeat: _array_repeat,
        ops.Unnest: _unnest,
        ops.Covariance: _covar,
        ops.Correlation: _corr,
        ops.BitwiseXor: _bitwise_op("#"),
        ops.Mode: _mode,
        ops.ApproxMedian: _median,
        ops.Median: _median,
        ops.Quantile: _quantile,
        ops.MultiQuantile: _quantile,
        ops.TimestampNow: lambda t, op: sa.literal_column(
            "CURRENT_TIMESTAMP", type_=t.get_sqla_type(op.dtype)
        ),
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.case(
                (arg.has_key(key), arg[key]), else_=default
            ),
            3,
        ),
        ops.MapContains: fixed_arity(pg.HSTORE.Comparator.has_key, 2),
        ops.MapKeys: unary(pg.HSTORE.Comparator.keys),
        ops.MapValues: unary(pg.HSTORE.Comparator.vals),
        ops.MapMerge: fixed_arity(operator.add, 2),
        ops.MapLength: unary(lambda arg: sa.func.cardinality(arg.keys())),
        ops.Map: fixed_arity(pg.hstore, 2),
        ops.ArgMin: _arg_min_max(sa.asc),
        ops.ArgMax: _arg_min_max(sa.desc),
        ops.ToJSONArray: unary(
            lambda arg: sa.case(
                (
                    sa.func.json_typeof(arg) == "array",
                    sa.func.array(
                        sa.select(
                            sa.func.json_array_elements(arg).column_valued()
                        ).scalar_subquery()
                    ),
                ),
                else_=sa.null(),
            )
        ),
        ops.ArrayStringJoin: fixed_arity(
            lambda sep, arr: sa.func.array_to_string(arr, sep), 2
        ),
        ops.Strip: unary(lambda arg: sa.func.trim(arg, string.whitespace)),
        ops.LStrip: unary(lambda arg: sa.func.ltrim(arg, string.whitespace)),
        ops.RStrip: unary(lambda arg: sa.func.rtrim(arg, string.whitespace)),
        ops.StartsWith: fixed_arity(lambda arg, prefix: arg.op("^@")(prefix), 2),
        ops.Arbitrary: _arbitrary,
        ops.StructColumn: _struct_column,
        ops.StructField: _struct_field,
        ops.First: reduction(sa.func.first),
        ops.Last: reduction(sa.func.last),
        ops.ExtractMicrosecond: fixed_arity(
            lambda arg: sa.extract("microsecond", arg) % 1_000_000, 1
        ),
        ops.Levenshtein: fixed_arity(sa.func.levenshtein, 2),
        ops.ArraySort: fixed_arity(_array_sort, 1),
        ops.ArrayIntersect: fixed_arity(
            lambda left, right: sa.func.array(
                sa.intersect(
                    sa.select(sa.func.unnest(left).column_valued()),
                    sa.select(sa.func.unnest(right).column_valued()),
                ).scalar_subquery()
            ),
            2,
        ),
        ops.ArrayRemove: fixed_arity(
            lambda left, right: sa.func.array(
                sa.except_(
                    sa.select(sa.func.unnest(left).column_valued()), sa.select(right)
                ).scalar_subquery()
            ),
            2,
        ),
        ops.ArrayUnion: fixed_arity(
            lambda left, right: sa.func.array(
                sa.union(
                    sa.select(sa.func.unnest(left).column_valued()),
                    sa.select(sa.func.unnest(right).column_valued()),
                ).scalar_subquery()
            ),
            2,
        ),
        ops.ArrayDistinct: fixed_arity(
            lambda arg: sa.case(
                (arg.is_(sa.null()), sa.null()),
                else_=sa.func.array(
                    sa.select(
                        sa.distinct(sa.func.unnest(arg).column_valued())
                    ).scalar_subquery()
                ),
            ),
            1,
        ),
        ops.ArrayPosition: fixed_arity(_array_position, 2),
        ops.ArrayMap: _array_map,
        ops.ArrayFilter: _array_filter,
    }
)
