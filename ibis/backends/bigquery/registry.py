"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import numpy as np
import sqlglot as sg
from multipledispatch import Dispatcher

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.registry import (
    fixed_arity,
    helpers,
    operation_registry,
    reduction,
    unary,
)
from ibis.backends.base.sql.registry.literal import _string_literal_format
from ibis.backends.base.sql.registry.main import table_array_view
from ibis.backends.bigquery.datatypes import BigQueryType
from ibis.common.temporal import DateUnit, IntervalUnit, TimeUnit

if TYPE_CHECKING:
    from ibis.backends.base.sql import compiler


def _extract_field(sql_attr):
    def extract_field_formatter(translator, op):
        arg = translator.translate(op.args[0])
        if sql_attr == "epochseconds":
            return f"UNIX_SECONDS({arg})"
        else:
            return f"EXTRACT({sql_attr} from {arg})"

    return extract_field_formatter


bigquery_cast = Dispatcher("bigquery_cast")


@bigquery_cast.register(str, dt.Timestamp, dt.Integer)
def bigquery_cast_timestamp_to_integer(compiled_arg, from_, to):
    """Convert TIMESTAMP to INT64 (seconds since Unix epoch)."""
    return f"UNIX_MICROS({compiled_arg})"


@bigquery_cast.register(str, dt.Integer, dt.Timestamp)
def bigquery_cast_integer_to_timestamp(compiled_arg, from_, to):
    """Convert INT64 (seconds since Unix epoch) to Timestamp."""
    return f"TIMESTAMP_SECONDS({compiled_arg})"


@bigquery_cast.register(str, dt.Interval, dt.Integer)
def bigquery_cast_interval_to_integer(compiled_arg, from_, to):
    if from_.unit in {IntervalUnit.WEEK, IntervalUnit.QUARTER, IntervalUnit.NANOSECOND}:
        raise com.UnsupportedOperationError(
            f"BigQuery does not allow extracting date part `{from_.unit}` from intervals"
        )

    return f"EXTRACT({from_.resolution.upper()} from {compiled_arg})"


@bigquery_cast.register(str, dt.Floating, dt.Integer)
def bigquery_cast_floating_to_integer(compiled_arg, from_, to):
    """Convert FLOAT64 to INT64 without rounding."""
    return f"CAST(TRUNC({compiled_arg}) AS INT64)"


@bigquery_cast.register(str, dt.DataType, dt.DataType)
def bigquery_cast_generate(compiled_arg, from_, to):
    """Cast to desired type."""
    sql_type = BigQueryType.from_ibis(to)
    return f"CAST({compiled_arg} AS {sql_type})"


@bigquery_cast.register(str, dt.DataType)
def bigquery_cast_generate_simple(compiled_arg, to):
    return bigquery_cast(compiled_arg, to, to)


def _cast(translator, op):
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)
    input_dtype = arg.dtype
    return bigquery_cast(arg_formatted, input_dtype, target_type)


def integer_to_timestamp(translator: compiler.ExprTranslator, op) -> str:
    """Interprets an integer as a timestamp."""
    arg = translator.translate(op.arg)
    unit = op.unit.short

    if unit == "s":
        return f"TIMESTAMP_SECONDS({arg})"
    elif unit == "ms":
        return f"TIMESTAMP_MILLIS({arg})"
    elif unit == "us":
        return f"TIMESTAMP_MICROS({arg})"
    elif unit == "ns":
        # Timestamps are represented internally as elapsed microseconds, so some
        # rounding is required if an integer represents nanoseconds.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#timestamp_type
        return f"TIMESTAMP_MICROS(CAST(ROUND({arg} / 1000) AS INT64))"

    raise NotImplementedError(f"cannot cast unit {op.unit}")


def _struct_field(translator, op):
    arg = translator.translate(op.arg)
    return f"{arg}.`{op.field}`"


def _struct_column(translator, op):
    cols = (
        f"{translator.translate(value)} AS {name}"
        for name, value, in zip(op.names, op.values)
    )
    return "STRUCT({})".format(", ".join(cols))


def _array_concat(translator, op):
    return "ARRAY_CONCAT({})".format(", ".join(map(translator.translate, op.arg)))


def _array_column(translator, op):
    return "[{}]".format(", ".join(map(translator.translate, op.cols)))


def _array_index(translator, op):
    # SAFE_OFFSET returns NULL if out of bounds
    arg = translator.translate(op.arg)
    index = translator.translate(op.index)
    return f"{arg}[SAFE_OFFSET({index})]"


def _array_contains(translator, op):
    arg = translator.translate(op.arg)
    other = translator.translate(op.other)
    name = util.gen_name("bq_arr")
    return f"(SELECT LOGICAL_OR({name} = {other}) FROM UNNEST({arg}) {name})"


def _array_position(translator, op):
    arg = translator.translate(op.arg)
    other = translator.translate(op.other)
    name = util.gen_name("bq_arr")
    idx = util.gen_name("bq_arr_idx")
    unnest = f"UNNEST({arg}) {name} WITH OFFSET AS {idx}"
    return f"COALESCE((SELECT {idx} FROM {unnest} WHERE {name} = {other} LIMIT 1), -1)"


def _array_remove(translator, op):
    arg = translator.translate(op.arg)
    other = translator.translate(op.other)
    name = util.gen_name("bq_arr")
    return f"ARRAY(SELECT {name} FROM UNNEST({arg}) {name} WHERE {name} <> {other})"


def _array_distinct(translator, op):
    arg = translator.translate(op.arg)
    name = util.gen_name("bq_arr")
    return f"ARRAY(SELECT DISTINCT {name} FROM UNNEST({arg}) {name})"


def _array_sort(translator, op):
    arg = translator.translate(op.arg)
    name = util.gen_name("bq_arr")
    return f"ARRAY(SELECT {name} FROM UNNEST({arg}) {name} ORDER BY {name})"


def _array_union(translator, op):
    left = translator.translate(op.left)
    right = translator.translate(op.right)

    lname = util.gen_name("bq_arr_left")
    rname = util.gen_name("bq_arr_right")

    left_expr = f"SELECT {lname} FROM UNNEST({left}) {lname}"
    right_expr = f"SELECT {rname} FROM UNNEST({right}) {rname}"

    return f"ARRAY({left_expr} UNION DISTINCT {right_expr})"


def _array_intersect(translator, op):
    left = translator.translate(op.left)
    right = translator.translate(op.right)

    lname = util.gen_name("bq_arr_left")
    rname = util.gen_name("bq_arr_right")

    left_expr = f"SELECT {lname} FROM UNNEST({left}) {lname}"
    right_expr = f"SELECT {rname} FROM UNNEST({right}) {rname}"

    return f"ARRAY({left_expr} INTERSECT DISTINCT {right_expr})"


def _array_zip(translator, op):
    arg = list(map(translator.translate, op.arg))
    lengths = ", ".join(map("ARRAY_LENGTH({}) - 1".format, arg))
    indices = f"UNNEST(GENERATE_ARRAY(0, GREATEST({lengths})))"
    idx = util.gen_name("bq_arr_idx")
    struct_fields = ", ".join(
        f"{arr}[SAFE_OFFSET({idx})] AS {name}"
        for name, arr in zip(op.dtype.value_type.names, arg)
    )
    return f"ARRAY(SELECT AS STRUCT {struct_fields} FROM {indices} {idx})"


def _array_map(translator, op):
    arg = translator.translate(op.arg)
    result = translator.translate(op.body)
    param = op.param
    return f"ARRAY(SELECT {result} FROM UNNEST({arg}) {param})"


def _array_filter(translator, op):
    arg = translator.translate(op.arg)
    result = translator.translate(op.body)
    param = op.param
    return f"ARRAY(SELECT {param} FROM UNNEST({arg}) {param} WHERE {result})"


def _hash(translator, op):
    arg_formatted = translator.translate(op.arg)
    return f"farm_fingerprint({arg_formatted})"


def _string_find(translator, op):
    haystack, needle, start, end = op.args

    if start is not None:
        raise NotImplementedError("start not implemented for string find")
    if end is not None:
        raise NotImplementedError("end not implemented for string find")

    return "STRPOS({}, {}) - 1".format(
        translator.translate(haystack), translator.translate(needle)
    )


def _regex_search(translator, op):
    arg = translator.translate(op.arg)
    regex = translator.translate(op.pattern)
    return f"REGEXP_CONTAINS({arg}, {regex})"


def _regex_extract(translator, op):
    arg = translator.translate(op.arg)
    regex = translator.translate(op.pattern)
    index = translator.translate(op.index)
    matches = f"REGEXP_CONTAINS({arg}, {regex})"
    # non-greedily match the regex's prefix so the regex can match as much as possible
    nonzero_index_replace = rf"REGEXP_REPLACE({arg}, CONCAT('.*?', {regex}, '.*'), CONCAT('\\', CAST({index} AS STRING)))"
    # zero index replacement means capture everything matched by the regex, so
    # we wrap the regex in an outer group
    zero_index_replace = (
        rf"REGEXP_REPLACE({arg}, CONCAT('.*?', CONCAT('(', {regex}, ')'), '.*'), '\\1')"
    )
    extract = f"IF({index} = 0, {zero_index_replace}, {nonzero_index_replace})"
    return f"IF({matches}, {extract}, NULL)"


def _regex_replace(translator, op):
    arg = translator.translate(op.arg)
    regex = translator.translate(op.pattern)
    replacement = translator.translate(op.replacement)
    return f"REGEXP_REPLACE({arg}, {regex}, {replacement})"


def _string_concat(translator, op):
    args = ", ".join(map(translator.translate, op.arg))
    return f"CONCAT({args})"


def _string_join(translator, op):
    sep, args = op.args
    return "ARRAY_TO_STRING([{}], {})".format(
        ", ".join(map(translator.translate, args)), translator.translate(sep)
    )


def _string_ascii(translator, op):
    arg = translator.translate(op.arg)
    return f"TO_CODE_POINTS({arg})[SAFE_OFFSET(0)]"


def _string_right(translator, op):
    arg, nchars = map(translator.translate, op.args)
    return f"SUBSTR({arg}, -LEAST(LENGTH({arg}), {nchars}))"


def _string_substring(translator, op):
    length = op.length
    if (length := getattr(length, "value", None)) is not None and length < 0:
        raise ValueError("Length parameter must be a non-negative value.")

    arg = translator.translate(op.arg)
    start = translator.translate(op.start)

    arg_length = f"LENGTH({arg})"
    if op.length is not None:
        suffix = f", {translator.translate(op.length)}"
    else:
        suffix = ""

    if_pos = f"SUBSTR({arg}, {start} + 1{suffix})"
    if_neg = f"SUBSTR({arg}, {arg_length} + {start} + 1{suffix})"
    return f"IF({start} >= 0, {if_pos}, {if_neg})"


def _log(translator, op):
    arg, base = op.args
    arg_formatted = translator.translate(arg)

    if base is None:
        return f"ln({arg_formatted})"

    base_formatted = translator.translate(base)
    return f"log({arg_formatted}, {base_formatted})"


def _sg_literal(val) -> str:
    return sg.exp.Literal(this=str(val), is_string=isinstance(val, str)).sql(
        dialect="bigquery"
    )


def _literal(t, op):
    dtype = op.dtype
    value = op.value

    if value is None:
        if not dtype.is_null():
            return f"CAST(NULL AS {BigQueryType.from_ibis(dtype)})"
        return "NULL"
    elif dtype.is_boolean():
        return str(value).upper()
    elif dtype.is_string() or dtype.is_inet() or dtype.is_macaddr():
        return _string_literal_format(t, op)
    elif dtype.is_decimal():
        if value.is_nan():
            return "CAST('NaN' AS FLOAT64)"
        elif value.is_infinite():
            prefix = "-" * value.is_signed()
            return f"CAST('{prefix}inf' AS FLOAT64)"
        else:
            return f"{BigQueryType.from_ibis(dtype)} '{value}'"
    elif dtype.is_uuid():
        return _sg_literal(str(value))
    elif dtype.is_numeric():
        if not np.isfinite(value):
            return f"CAST({str(value)!r} AS FLOAT64)"
        return _sg_literal(value)
    elif dtype.is_date():
        with contextlib.suppress(AttributeError):
            value = value.date()
        return f"DATE {_sg_literal(str(value))}"
    elif dtype.is_timestamp():
        typename = "DATETIME" if dtype.timezone is None else "TIMESTAMP"
        return f"{typename} {_sg_literal(str(value))}"
    elif dtype.is_time():
        # TODO: define extractors on TimeValue expressions
        return f"TIME {_sg_literal(str(value))}"
    elif dtype.is_binary():
        return repr(value)
    elif dtype.is_struct():
        cols = ", ".join(
            f"{t.translate(ops.Literal(value[name], dtype=typ))} AS `{name}`"
            for name, typ in dtype.items()
        )
        return f"STRUCT({cols})"
    elif dtype.is_array():
        val_type = dtype.value_type
        values = ", ".join(
            t.translate(ops.Literal(element, dtype=val_type)) for element in value
        )
        return f"[{values}]"
    elif dtype.is_interval():
        return f"INTERVAL {value} {dtype.resolution.upper()}"
    else:
        raise NotImplementedError(f"Unsupported type for BigQuery literal: {dtype}")


def _arbitrary(translator, op):
    arg, how, where = op.args

    if where is not None:
        arg = ops.IfElse(where, arg, ibis.NA)

    if how != "first":
        raise com.UnsupportedOperationError(
            f"{how!r} value not supported for arbitrary in BigQuery"
        )

    return f"ANY_VALUE({translator.translate(arg)})"


def _first(translator, op):
    arg = op.arg
    where = op.where

    if where is not None:
        arg = ops.IfElse(where, arg, ibis.NA)

    arg = translator.translate(arg)
    return f"ARRAY_AGG({arg} IGNORE NULLS)[SAFE_OFFSET(0)]"


def _last(translator, op):
    arg = op.arg
    where = op.where

    if where is not None:
        arg = ops.IfElse(where, arg, ibis.NA)

    arg = translator.translate(arg)
    return f"ARRAY_REVERSE(ARRAY_AGG({arg} IGNORE NULLS))[SAFE_OFFSET(0)]"


def _truncate(kind, units):
    def truncator(translator, op):
        arg, unit = op.args
        trans_arg = translator.translate(arg)
        if unit not in units:
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {arg.dtype} values to unit {unit!r}"
            )
        if unit.name == "WEEK":
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return f"{kind}_TRUNC({trans_arg}, {unit})"

    return truncator


# BigQuery doesn't support nanosecond intervals
_date_truncate = _truncate("DATE", DateUnit)
_time_truncate = _truncate("TIME", set(TimeUnit) - {TimeUnit.NANOSECOND})
_timestamp_truncate = _truncate(
    "TIMESTAMP", set(IntervalUnit) - {IntervalUnit.NANOSECOND}
)


def _date_binary(func):
    def _formatter(translator, op):
        arg, offset = op.left, op.right

        unit = offset.dtype.unit
        if not unit.is_date():
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
            )

        formatted_arg = translator.translate(arg)
        formatted_offset = translator.translate(offset)
        return f"{func}({formatted_arg}, {formatted_offset})"

    return _formatter


def _timestamp_binary(func):
    def _formatter(translator, op):
        arg, offset = op.left, op.right

        unit = offset.dtype.unit
        if unit == IntervalUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
            )

        if unit.is_date():
            try:
                offset = offset.to_expr().to_unit("h").op()
            except ValueError:
                raise com.UnsupportedOperationError(
                    f"BigQuery does not allow binary operation {func} with INTERVAL offset {unit}"
                )

        formatted_arg = translator.translate(arg)
        formatted_offset = translator.translate(offset)
        return f"{func}({formatted_arg}, {formatted_offset})"

    return _formatter


def _geo_boundingbox(dimension_name):
    def _formatter(translator, op):
        geog = op.args[0]
        geog_formatted = translator.translate(geog)
        return f"ST_BOUNDINGBOX({geog_formatted}).{dimension_name}"

    return _formatter


def _geo_simplify(translator, op):
    geog, tolerance, preserve_collapsed = op.args
    if preserve_collapsed.value:
        raise com.UnsupportedOperationError(
            "BigQuery simplify does not support preserving collapsed geometries, "
            "must pass preserve_collapsed=False"
        )
    geog, tolerance = map(translator.translate, (geog, tolerance))
    return f"ST_SIMPLIFY({geog}, {tolerance})"


STRFTIME_FORMAT_FUNCTIONS = {
    dt.date: "DATE",
    dt.time: "TIME",
    dt.Timestamp(timezone=None): "DATETIME",
    dt.Timestamp(timezone="UTC"): "TIMESTAMP",
}


def bigquery_day_of_week_index(t, op):
    """Convert timestamp to day-of-week integer."""
    arg = op.args[0]
    arg_formatted = t.translate(arg)
    return f"MOD(EXTRACT(DAYOFWEEK FROM {arg_formatted}) + 5, 7)"


def bigquery_day_of_week_name(t, op):
    """Convert timestamp to day-of-week name."""
    return f"INITCAP(CAST({t.translate(op.arg)} AS STRING FORMAT 'DAY'))"


def bigquery_compiles_divide(t, op):
    """Floating point division."""
    return f"IEEE_DIVIDE({t.translate(op.left)}, {t.translate(op.right)})"


def compiles_strftime(translator, op):
    """Timestamp formatting."""
    arg = op.arg
    format_str = op.format_str
    arg_type = arg.dtype
    strftime_format_func_name = STRFTIME_FORMAT_FUNCTIONS[arg_type]
    fmt_string = translator.translate(format_str)
    arg_formatted = translator.translate(arg)
    if isinstance(arg_type, dt.Timestamp) and arg_type.timezone is None:
        return f"FORMAT_{strftime_format_func_name}({fmt_string}, {arg_formatted})"
    elif isinstance(arg_type, dt.Timestamp):
        return "FORMAT_{}({}, {}, {!r})".format(
            strftime_format_func_name,
            fmt_string,
            arg_formatted,
            arg_type.timezone,
        )
    else:
        return f"FORMAT_{strftime_format_func_name}({fmt_string}, {arg_formatted})"


def compiles_string_to_timestamp(translator, op):
    """Timestamp parsing."""
    fmt_string = translator.translate(op.format_str)
    arg_formatted = translator.translate(op.arg)
    return f"PARSE_TIMESTAMP({fmt_string}, {arg_formatted})"


def compiles_floor(t, op):
    bigquery_type = BigQueryType.from_ibis(op.dtype)
    arg = op.arg
    return f"CAST(FLOOR({t.translate(arg)}) AS {bigquery_type})"


def compiles_approx(translator, op):
    arg = op.arg
    where = op.where

    if where is not None:
        arg = ops.IfElse(where, arg, ibis.NA)

    return f"APPROX_QUANTILES({translator.translate(arg)}, 2)[OFFSET(1)]"


def compiles_covar_corr(func):
    def translate(translator, op):
        left = op.left
        right = op.right

        if (where := op.where) is not None:
            left = ops.IfElse(where, left, None)
            right = ops.IfElse(where, right, None)

        left = translator.translate(
            ops.Cast(left, dt.int64) if left.dtype.is_boolean() else left
        )
        right = translator.translate(
            ops.Cast(right, dt.int64) if right.dtype.is_boolean() else right
        )
        return f"{func}({left}, {right})"

    return translate


def _covar(translator, op):
    how = op.how[:4].upper()
    assert how in ("POP", "SAMP"), 'how not in ("POP", "SAMP")'
    return compiles_covar_corr(f"COVAR_{how}")(translator, op)


def _corr(translator, op):
    if (how := op.how) == "sample":
        raise ValueError(f"Correlation with how={how!r} is not supported.")
    return compiles_covar_corr("CORR")(translator, op)


def _identical_to(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return f"{left} IS NOT DISTINCT FROM {right}"


def _floor_divide(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return bigquery_cast(f"FLOOR(IEEE_DIVIDE({left}, {right}))", op.dtype)


def _log2(t, op):
    return f"LOG({t.translate(op.arg)}, 2)"


def _is_nan(t, op):
    return f"IS_NAN({t.translate(op.arg)})"


def _is_inf(t, op):
    return f"IS_INF({t.translate(op.arg)})"


def _array_agg(t, op):
    arg = op.arg
    if (where := op.where) is not None:
        arg = ops.IfElse(where, arg, ibis.NA)
    return f"ARRAY_AGG({t.translate(arg)} IGNORE NULLS)"


def _arg_min_max(sort_dir: Literal["ASC", "DESC"]):
    def translate(t, op: ops.ArgMin | ops.ArgMax) -> str:
        arg = op.arg
        if (where := op.where) is not None:
            arg = ops.IfElse(where, arg, None)
        arg = t.translate(arg)
        key = t.translate(op.key)
        return f"ARRAY_AGG({arg} IGNORE NULLS ORDER BY {key} {sort_dir} LIMIT 1)[SAFE_OFFSET(0)]"

    return translate


def _array_repeat(t, op):
    start = step = 1
    times = t.translate(op.times)
    arg = t.translate(op.arg)
    array_length = f"ARRAY_LENGTH({arg})"
    stop = f"GREATEST({times}, 0) * {array_length}"
    idx = f"COALESCE(NULLIF(MOD(i, {array_length}), 0), {array_length})"
    series = f"GENERATE_ARRAY({start}, {stop}, {step})"
    return f"ARRAY(SELECT {arg}[SAFE_ORDINAL({idx})] FROM UNNEST({series}) AS i)"


def _neg_idx_to_pos(array, idx):
    return f"IF({idx} < 0, ARRAY_LENGTH({array}) + {idx}, {idx})"


def _array_slice(t, op):
    arg = t.translate(op.arg)
    cond = [f"index >= {_neg_idx_to_pos(arg, t.translate(op.start))}"]
    if stop := op.stop:
        cond.append(f"index < {_neg_idx_to_pos(arg, t.translate(stop))}")
    return (
        f"ARRAY("
        f"SELECT el "
        f"FROM UNNEST({arg}) AS el WITH OFFSET index "
        f"WHERE {' AND '.join(cond)}"
        f")"
    )


def _capitalize(t, op):
    arg = t.translate(op.arg)
    return f"CONCAT(UPPER(SUBSTR({arg}, 1, 1)), LOWER(SUBSTR({arg}, 2)))"


def _nth_value(t, op):
    arg = t.translate(op.arg)

    if not isinstance(nth_op := op.nth, ops.Literal):
        raise TypeError(f"Bigquery nth must be a literal; got {type(op.nth)}")

    return f"NTH_VALUE({arg}, {nth_op.value + 1})"


def _interval_multiply(t, op):
    if isinstance(op.left, ops.Literal) and isinstance(op.right, ops.Literal):
        value = op.left.value * op.right.value
        literal = ops.Literal(value, op.left.dtype)
        return t.translate(literal)

    left, right = t.translate(op.left), t.translate(op.right)
    unit = op.left.dtype.resolution.upper()
    return f"INTERVAL EXTRACT({unit} from {left}) * {right} {unit}"


def table_column(translator, op):
    """Override column references to adjust names for BigQuery."""
    quoted_name = translator._gen_valid_name(
        helpers.quote_identifier(op.name, force=True)
    )

    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if translator.permit_subquery and ctx.is_foreign_expr(op.table):
        # TODO(kszucs): avoid the expression roundtrip
        proj_expr = op.table.to_expr().select([op.name]).to_array().op()
        return table_array_view(translator, proj_expr)

    alias = ctx.get_ref(op.table, search_parents=True)
    if alias is not None:
        quoted_name = f"{alias}.{quoted_name}"

    return quoted_name


def _count_distinct_star(t, op):
    raise com.UnsupportedOperationError(
        "BigQuery doesn't support COUNT(DISTINCT ...) with multiple columns"
    )


def _time_delta(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return f"TIME_DIFF({left}, {right}, {op.part.value.upper()})"


def _date_delta(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return f"DATE_DIFF({left}, {right}, {op.part.value.upper()})"


def _timestamp_delta(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    left_tz = op.left.dtype.timezone
    right_tz = op.right.dtype.timezone
    args = f"{left}, {right}, {op.part.value.upper()}"
    if left_tz is None and right_tz is None:
        return f"DATETIME_DIFF({args})"
    elif left_tz is not None and right_tz is not None:
        return f"TIMESTAMP_DIFF({args})"
    else:
        raise NotImplementedError(
            "timestamp difference with mixed timezone/timezoneless values is not implemented"
        )


OPERATION_REGISTRY = {
    **operation_registry,
    # Literal
    ops.Literal: _literal,
    # Logical
    ops.Any: reduction("LOGICAL_OR"),
    ops.All: reduction("LOGICAL_AND"),
    ops.NullIf: fixed_arity("NULLIF", 2),
    # Reductions
    ops.ApproxMedian: compiles_approx,
    ops.Covariance: _covar,
    ops.Correlation: _corr,
    # Math
    ops.Divide: bigquery_compiles_divide,
    ops.Floor: compiles_floor,
    ops.Modulus: fixed_arity("MOD", 2),
    ops.Sign: unary("SIGN"),
    ops.BitwiseNot: lambda t, op: f"~ {t.translate(op.arg)}",
    ops.BitwiseXor: lambda t, op: f"{t.translate(op.left)} ^ {t.translate(op.right)}",
    ops.BitwiseOr: lambda t, op: f"{t.translate(op.left)} | {t.translate(op.right)}",
    ops.BitwiseAnd: lambda t, op: f"{t.translate(op.left)} & {t.translate(op.right)}",
    ops.BitwiseLeftShift: lambda t, op: f"{t.translate(op.left)} << {t.translate(op.right)}",
    ops.BitwiseRightShift: lambda t, op: f"{t.translate(op.left)} >> {t.translate(op.right)}",
    # Temporal functions
    ops.Date: unary("DATE"),
    ops.DateFromYMD: fixed_arity("DATE", 3),
    ops.DateAdd: _date_binary("DATE_ADD"),
    ops.DateSub: _date_binary("DATE_SUB"),
    ops.DateTruncate: _date_truncate,
    ops.DayOfWeekIndex: bigquery_day_of_week_index,
    ops.DayOfWeekName: bigquery_day_of_week_name,
    ops.ExtractEpochSeconds: _extract_field("epochseconds"),
    ops.ExtractYear: _extract_field("year"),
    ops.ExtractQuarter: _extract_field("quarter"),
    ops.ExtractMonth: _extract_field("month"),
    ops.ExtractWeekOfYear: _extract_field("isoweek"),
    ops.ExtractDay: _extract_field("day"),
    ops.ExtractDayOfYear: _extract_field("dayofyear"),
    ops.ExtractHour: _extract_field("hour"),
    ops.ExtractMinute: _extract_field("minute"),
    ops.ExtractSecond: _extract_field("second"),
    ops.ExtractMicrosecond: _extract_field("microsecond"),
    ops.ExtractMillisecond: _extract_field("millisecond"),
    ops.Strftime: compiles_strftime,
    ops.StringToTimestamp: compiles_string_to_timestamp,
    ops.Time: unary("TIME"),
    ops.TimeFromHMS: fixed_arity("TIME", 3),
    ops.TimeTruncate: _time_truncate,
    ops.TimestampAdd: _timestamp_binary("TIMESTAMP_ADD"),
    ops.TimestampFromUNIX: integer_to_timestamp,
    ops.TimestampFromYMDHMS: fixed_arity("DATETIME", 6),
    ops.TimestampNow: fixed_arity("CURRENT_TIMESTAMP", 0),
    ops.TimestampSub: _timestamp_binary("TIMESTAMP_SUB"),
    ops.TimestampTruncate: _timestamp_truncate,
    ops.IntervalMultiply: _interval_multiply,
    ops.Hash: _hash,
    ops.StringReplace: fixed_arity("REPLACE", 3),
    ops.StringSplit: fixed_arity("SPLIT", 2),
    ops.StringConcat: _string_concat,
    ops.StringJoin: _string_join,
    ops.StringAscii: _string_ascii,
    ops.StringFind: _string_find,
    ops.Substring: _string_substring,
    ops.StrRight: _string_right,
    ops.Capitalize: _capitalize,
    ops.Translate: fixed_arity("TRANSLATE", 3),
    ops.Repeat: fixed_arity("REPEAT", 2),
    ops.RegexSearch: _regex_search,
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: _regex_replace,
    ops.GroupConcat: reduction("STRING_AGG"),
    ops.Cast: _cast,
    ops.StructField: _struct_field,
    ops.StructColumn: _struct_column,
    ops.ArrayCollect: _array_agg,
    ops.ArrayConcat: _array_concat,
    ops.ArrayColumn: _array_column,
    ops.ArrayIndex: _array_index,
    ops.ArrayLength: unary("ARRAY_LENGTH"),
    ops.ArrayRepeat: _array_repeat,
    ops.ArraySlice: _array_slice,
    ops.ArrayContains: _array_contains,
    ops.ArrayPosition: _array_position,
    ops.ArrayRemove: _array_remove,
    ops.ArrayDistinct: _array_distinct,
    ops.ArraySort: _array_sort,
    ops.ArrayUnion: _array_union,
    ops.ArrayIntersect: _array_intersect,
    ops.ArrayZip: _array_zip,
    ops.ArrayMap: _array_map,
    ops.ArrayFilter: _array_filter,
    ops.Log: _log,
    ops.Log2: _log2,
    ops.Arbitrary: _arbitrary,
    ops.First: _first,
    ops.Last: _last,
    # Geospatial Columnar
    ops.GeoUnaryUnion: unary("ST_UNION_AGG"),
    # Geospatial
    ops.GeoArea: unary("ST_AREA"),
    ops.GeoAsBinary: unary("ST_ASBINARY"),
    ops.GeoAsText: unary("ST_ASTEXT"),
    ops.GeoAzimuth: fixed_arity("ST_AZIMUTH", 2),
    ops.GeoBuffer: fixed_arity("ST_BUFFER", 2),
    ops.GeoCentroid: unary("ST_CENTROID"),
    ops.GeoContains: fixed_arity("ST_CONTAINS", 2),
    ops.GeoCovers: fixed_arity("ST_COVERS", 2),
    ops.GeoCoveredBy: fixed_arity("ST_COVEREDBY", 2),
    ops.GeoDWithin: fixed_arity("ST_DWITHIN", 3),
    ops.GeoDifference: fixed_arity("ST_DIFFERENCE", 2),
    ops.GeoDisjoint: fixed_arity("ST_DISJOINT", 2),
    ops.GeoDistance: fixed_arity("ST_DISTANCE", 2),
    ops.GeoEndPoint: unary("ST_ENDPOINT"),
    ops.GeoEquals: fixed_arity("ST_EQUALS", 2),
    ops.GeoGeometryType: unary("ST_GEOMETRYTYPE"),
    ops.GeoIntersection: fixed_arity("ST_INTERSECTION", 2),
    ops.GeoIntersects: fixed_arity("ST_INTERSECTS", 2),
    ops.GeoLength: unary("ST_LENGTH"),
    ops.GeoMaxDistance: fixed_arity("ST_MAXDISTANCE", 2),
    ops.GeoNPoints: unary("ST_NUMPOINTS"),
    ops.GeoPerimeter: unary("ST_PERIMETER"),
    ops.GeoPoint: fixed_arity("ST_GEOGPOINT", 2),
    ops.GeoPointN: fixed_arity("ST_POINTN", 2),
    ops.GeoSimplify: _geo_simplify,
    ops.GeoStartPoint: unary("ST_STARTPOINT"),
    ops.GeoTouches: fixed_arity("ST_TOUCHES", 2),
    ops.GeoUnion: fixed_arity("ST_UNION", 2),
    ops.GeoWithin: fixed_arity("ST_WITHIN", 2),
    ops.GeoX: unary("ST_X"),
    ops.GeoXMax: _geo_boundingbox("xmax"),
    ops.GeoXMin: _geo_boundingbox("xmin"),
    ops.GeoY: unary("ST_Y"),
    ops.GeoYMax: _geo_boundingbox("ymax"),
    ops.GeoYMin: _geo_boundingbox("ymin"),
    ops.BitAnd: reduction("BIT_AND"),
    ops.BitOr: reduction("BIT_OR"),
    ops.BitXor: reduction("BIT_XOR"),
    ops.ApproxCountDistinct: reduction("APPROX_COUNT_DISTINCT"),
    ops.ApproxMedian: compiles_approx,
    ops.IdenticalTo: _identical_to,
    ops.FloorDivide: _floor_divide,
    ops.IsNan: _is_nan,
    ops.IsInf: _is_inf,
    ops.ArgMin: _arg_min_max("ASC"),
    ops.ArgMax: _arg_min_max("DESC"),
    ops.Pi: lambda *_: "ACOS(-1)",
    ops.E: lambda *_: "EXP(1)",
    ops.RandomScalar: fixed_arity("RAND", 0),
    ops.NthValue: _nth_value,
    ops.JSONGetItem: lambda t, op: f"{t.translate(op.arg)}[{t.translate(op.index)}]",
    ops.ArrayStringJoin: lambda t, op: f"ARRAY_TO_STRING({t.translate(op.arg)}, {t.translate(op.sep)})",
    ops.StartsWith: fixed_arity("STARTS_WITH", 2),
    ops.EndsWith: fixed_arity("ENDS_WITH", 2),
    ops.TableColumn: table_column,
    ops.CountDistinctStar: _count_distinct_star,
    ops.Argument: lambda _, op: op.name,
    ops.Unnest: unary("UNNEST"),
    ops.TimeDelta: _time_delta,
    ops.DateDelta: _date_delta,
    ops.TimestampDelta: _timestamp_delta,
}

_invalid_operations = {
    ops.FindInSet,
    ops.DateDiff,
    ops.TimestampDiff,
    ops.ExtractAuthority,
    ops.ExtractFile,
    ops.ExtractFragment,
    ops.ExtractHost,
    ops.ExtractPath,
    ops.ExtractProtocol,
    ops.ExtractQuery,
    ops.ExtractUserInfo,
}

OPERATION_REGISTRY = {
    k: v for k, v in OPERATION_REGISTRY.items() if k not in _invalid_operations
}
