import sqlalchemy as sa

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type
from ibis.backends.base.sql.alchemy.registry import _literal as _alchemy_literal
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.postgres.registry import _corr, _covar

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)


def _array(t, elements):
    return t.translate(ibis.array(elements).op())


def _literal(t, op):
    value = op.value
    dtype = op.output_dtype

    if dtype.is_struct():
        return sa.cast(sa.func.row(*value.values()), to_sqla_type(dtype))
    elif dtype.is_map():
        return sa.func.map(_array(t, value.keys()), _array(t, value.values()))
    return _alchemy_literal(t, op)


def _arbitrary(t, op):
    if op.how == "heavy":
        raise ValueError('Trino does not support how="heavy"')
    return reduction(sa.func.arbitrary)(t, op)


def _json_get_item(t, op):
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    fmt = "%d" if op.index.output_dtype.is_integer() else '"%s"'
    return sa.func.json_extract(arg, sa.func.format(f"$[{fmt}]", index))


def _group_concat(t, op):
    if not isinstance(op.sep, ops.Literal):
        raise com.IbisTypeError("Trino group concat separator must be a literal value")

    arg = sa.func.array_agg(t.translate(op.arg))
    if (where := op.where) is not None:
        arg = arg.filter(t.translate(where))
    return sa.func.array_join(arg, t.translate(op.sep))


def _array_column(t, op):
    args = ", ".join(
        str(t.translate(arg).compile(compile_kwargs={"literal_binds": True}))
        for arg in op.cols
    )
    return sa.literal_column(f"ARRAY[{args}]", type_=to_sqla_type(op.output_dtype))


_truncate_precisions = {
    # ms unit is not yet officially documented in Trino's public documentation,
    # but it just works.
    'ms': 'millisecond',
    's': 'second',
    'm': 'minute',
    'h': 'hour',
    'D': 'day',
    'W': 'week',
    'M': 'month',
    'Q': 'quarter',
    'Y': 'year',
}


def _timestamp_truncate(t, op):
    sa_arg = t.translate(op.arg)
    try:
        precision = _truncate_precisions[op.unit]
    except KeyError:
        raise com.UnsupportedOperationError(f'Unsupported truncate unit {op.unit!r}')
    return sa.func.date_trunc(precision, sa_arg)


def _timestamp_from_unix(t, op):
    arg, unit = op.args
    arg = t.translate(arg)

    if unit == "ms":
        return sa.func.from_unixtime(arg / 1_000)
    elif unit == "s":
        return sa.func.from_unixtime(arg)
    elif unit == "us":
        return sa.func.from_unixtime_nanos((arg - arg % 1_000_000) * 1_000)
    elif unit == "ns":
        return sa.func.from_unixtime_nanos(arg - (arg % 1_000_000_000))
    else:
        raise ValueError(f"{unit!r} unit is not supported!")


def _neg_idx_to_pos(array, idx):
    if_ = getattr(sa.func, "if")
    arg_length = sa.func.cardinality(array)
    return if_(idx < 0, arg_length + sa.func.greatest(idx, -arg_length), idx)


def _array_slice(t, op):
    arg = t.translate(op.arg)

    arg_length = sa.func.cardinality(arg)

    if (start := op.start) is None:
        start = 0
    else:
        start = sa.func.least(arg_length, _neg_idx_to_pos(arg, t.translate(start)))

    if (stop := op.stop) is None:
        stop = arg_length
    else:
        stop = _neg_idx_to_pos(arg, t.translate(stop))

    length = stop - start
    return sa.func.slice(arg, start + 1, length, type_=arg.type)


_PARSE_URL_FUNCS = {
    "PROTOCOL": lambda arg, _: sa.func.url_extract_protocol(arg),
    "HOST": lambda arg, _: sa.func.url_extract_host(arg),
    "PATH": lambda arg, _: sa.func.url_extract_path(arg),
    "REF": lambda arg, _: sa.func.url_extract_fragment(arg),
    "FILE": lambda arg, _: sa.func.concat_ws(
        "?",
        sa.func.nullif(sa.func.url_extract_path(arg), ""),
        sa.func.nullif(sa.func.url_extract_query(arg), ""),
    ),
    "QUERY": lambda arg, key: (
        sa.func.url_extract_parameter(arg, key)
        if key is not None
        else sa.func.url_extract_query(arg)
    ),
}


def _parse_url(t, op):
    if (func := _PARSE_URL_FUNCS.get(extract := op.extract)) is None:
        raise ValueError(f"`{extract}` is not supported in the Trino backend")

    return sa.func.nullif(
        func(t.translate(op.arg), key if (key := op.key) is None else t.translate(key)),
        "",
    )


operation_registry.update(
    {
        # conditional expressions
        # static checks are not happy with using "if" as a property
        ops.Where: fixed_arity(getattr(sa.func, 'if'), 3),
        # boolean reductions
        ops.Any: unary(sa.func.bool_or),
        ops.All: unary(sa.func.bool_and),
        ops.NotAny: unary(lambda x: sa.not_(sa.func.bool_or(x))),
        ops.NotAll: unary(lambda x: sa.not_(sa.func.bool_and(x))),
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        # array ops
        ops.Correlation: _corr,
        ops.Covariance: _covar,
        ops.ExtractMillisecond: unary(sa.func.millisecond),
        ops.Arbitrary: _arbitrary,
        ops.ApproxCountDistinct: reduction(sa.func.approx_distinct),
        ops.ParseURL: _parse_url,
        ops.RegexExtract: fixed_arity(sa.func.regexp_extract, 3),
        ops.RegexReplace: fixed_arity(sa.func.regexp_replace, 3),
        ops.RegexSearch: fixed_arity(
            lambda arg, pattern: sa.func.regexp_position(arg, pattern) != -1, 2
        ),
        ops.GroupConcat: _group_concat,
        ops.BitAnd: reduction(sa.func.bitwise_and_agg),
        ops.BitOr: reduction(sa.func.bitwise_or_agg),
        ops.BitwiseAnd: fixed_arity(sa.func.bitwise_and, 2),
        ops.BitwiseOr: fixed_arity(sa.func.bitwise_or, 2),
        ops.BitwiseXor: fixed_arity(sa.func.bitwise_xor, 2),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitwise_left_shift, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitwise_right_shift, 2),
        ops.BitwiseNot: unary(sa.func.bitwise_not),
        ops.ArrayCollect: reduction(sa.func.array_agg),
        ops.ArrayConcat: fixed_arity(sa.func.concat, 2),
        ops.ArrayLength: unary(sa.func.cardinality),
        ops.ArrayIndex: fixed_arity(
            lambda arg, index: sa.func.element_at(arg, index + 1), 2
        ),
        ops.ArrayColumn: _array_column,
        ops.ArrayRepeat: fixed_arity(
            lambda arg, times: sa.func.flatten(sa.func.repeat(arg, times)), 2
        ),
        ops.ArraySlice: _array_slice,
        ops.JSONGetItem: _json_get_item,
        ops.ExtractDayOfYear: unary(sa.func.day_of_year),
        ops.ExtractWeekOfYear: unary(sa.func.week_of_year),
        ops.DayOfWeekIndex: fixed_arity(
            lambda arg: sa.cast(
                sa.cast(sa.func.day_of_week(arg) + 6, sa.SMALLINT) % 7, sa.SMALLINT
            ),
            1,
        ),
        ops.DayOfWeekName: fixed_arity(lambda arg: sa.func.date_format(arg, "%W"), 1),
        ops.ExtractEpochSeconds: unary(sa.func.to_unixtime),
        ops.Translate: fixed_arity(sa.func.translate, 3),
        ops.Capitalize: fixed_arity(
            lambda arg: sa.func.concat(
                sa.func.upper(sa.func.substring(arg, 1, 2)), sa.func.substring(arg, 2)
            ),
            1,
        ),
        ops.StrRight: fixed_arity(lambda arg, nchars: sa.func.substr(arg, -nchars), 2),
        ops.StringSplit: fixed_arity(sa.func.split, 2),
        ops.Repeat: fixed_arity(
            lambda value, count: sa.func.array_join(sa.func.repeat(value, count), ''), 2
        ),
        ops.DateTruncate: _timestamp_truncate,
        ops.TimestampTruncate: _timestamp_truncate,
        ops.DateFromYMD: fixed_arity(
            lambda y, m, d: sa.func.from_iso8601_date(
                sa.func.format('%04d-%02d-%02d', y, m, d)
            ),
            3,
        ),
        ops.TimeFromHMS: fixed_arity(
            lambda h, m, s: sa.cast(sa.func.format('%02d:%02d:%02d', h, m, s), sa.TIME),
            3,
        ),
        ops.TimestampFromYMDHMS: fixed_arity(
            lambda y, mo, d, h, m, s: sa.func.from_iso8601_timestamp(
                sa.func.format('%04d-%02d-%02dT%02d:%02d:%02d', y, mo, d, h, m, s)
            ),
            6,
        ),
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.StringToTimestamp: fixed_arity(sa.func.date_parse, 2),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.StructField: (lambda t, op: t.translate(op.arg).op(".")(sa.text(op.field))),
        ops.StructColumn: (
            lambda t, op: sa.cast(
                sa.func.row(*map(t.translate, op.values)), to_sqla_type(op.output_dtype)
            )
        ),
        ops.Literal: _literal,
        ops.MapLength: fixed_arity(sa.func.cardinality, 1),
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.func.coalesce(
                sa.func.element_at(arg, key), default
            ),
            3,
        ),
        ops.MapKeys: fixed_arity(sa.func.map_keys, 1),
        ops.MapValues: fixed_arity(sa.func.map_values, 1),
        ops.Map: fixed_arity(sa.func.map, 2),
        ops.MapMerge: fixed_arity(sa.func.map_concat, 2),
        ops.MapContains: fixed_arity(
            lambda arg, key: sa.func.contains(sa.func.map_keys(arg), key), 2
        ),
    }
)
