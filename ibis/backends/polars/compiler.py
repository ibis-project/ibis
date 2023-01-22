from __future__ import annotations

import calendar
import functools
import math
import operator
from typing import Mapping

import numpy as np
import pandas as pd
import polars as pl
from packaging.version import parse as vparse

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.polars.datatypes import to_polars_type


def _assert_literal(op):
    # TODO(kszucs): broadcast and apply UDF on two columns using concat_list
    # TODO(kszucs): better error message
    if not isinstance(op, ops.Literal):
        raise com.UnsupportedArgumentError(
            f"Polars does not support columnar argument {op.name}"
        )


@functools.singledispatch
def translate(expr):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op):
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate.register(ops.DatabaseTable)
def table(op):
    return op.source._tables[op.name]


@translate.register(ops.InMemoryTable)
def pandas_in_memory_table(op):
    lf = pl.from_pandas(op.data.to_frame()).lazy()

    columns = []
    for name, current_dtype in sch.infer(lf).items():
        desired_dtype = op.schema[name]
        if current_dtype != desired_dtype:
            typ = to_polars_type(desired_dtype)
            columns.append(pl.col(name).cast(typ))

    if columns:
        return lf.with_columns(columns)
    else:
        return lf


@translate.register(ops.Alias)
def alias(op):
    arg = translate(op.arg)
    return arg.alias(op.name)


def _make_duration(value, dtype):
    kwargs = {f"{dtype.resolution}s": value}
    return pl.duration(**kwargs)


@translate.register(ops.Literal)
def literal(op):
    value = op.value
    dtype = op.dtype

    if dtype.is_array():
        value = pl.Series("", value)
        typ = to_polars_type(dtype)
        return pl.lit(value, dtype=typ).list()
    elif dtype.is_struct():
        values = [
            pl.lit(v, dtype=to_polars_type(dtype[k])).alias(k) for k, v in value.items()
        ]
        return pl.struct(values)
    elif dtype.is_interval():
        return _make_duration(value, dtype)
    elif dtype.is_null():
        return pl.lit(value)
    elif dtype.is_binary():
        raise NotImplementedError("Binary literals do not work in polars")
    else:
        typ = to_polars_type(dtype)
        return pl.lit(op.value, dtype=typ)


_TIMESTAMP_SCALE_TO_UNITS = {
    0: "s",
    1: "ms",
    2: "ms",
    3: "ms",
    4: "us",
    5: "us",
    6: "us",
    7: "ns",
    8: "ns",
    9: "ns",
}


@translate.register(ops.Cast)
def cast(op):
    arg = translate(op.arg)
    dtype = op.arg.output_dtype
    to = op.to

    if to.is_interval():
        return _make_duration(arg, to)
    elif to.is_date():
        if dtype.is_string():
            return arg.str.strptime(pl.Date, "%Y-%m-%d")
    elif to.is_timestamp():
        time_zone = to.timezone
        time_unit = _TIMESTAMP_SCALE_TO_UNITS.get(to.scale, "us")

        if dtype.is_integer():
            typ = pl.Datetime(time_unit="us", time_zone=time_zone)
            arg = (arg * 1_000_000).cast(typ)
            if time_unit != "us":
                arg = arg.dt.truncate(f"1{time_unit}")
            return arg.alias(op.name)
        elif dtype.is_string():
            typ = pl.Datetime(time_unit=time_unit, time_zone=time_zone)
            arg = arg.str.strptime(typ)
            if time_unit == "s":
                return arg.dt.truncate("1s")
            return arg

    typ = to_polars_type(to)
    return arg.cast(typ)


@translate.register(ops.TableColumn)
def column(op):
    return pl.col(op.name)


@translate.register(ops.SortKey)
def sort_key(op):
    arg = translate(op.expr)
    return arg.sort(reverse=op.descending)


@translate.register(ops.Selection)
def selection(op):
    lf = translate(op.table)

    if op.predicates:
        predicates = map(translate, op.predicates)
        predicate = functools.reduce(operator.and_, predicates)
        lf = lf.filter(predicate)

    selections = []
    for arg in op.selections:
        if isinstance(arg, ops.TableNode):
            for name in arg.schema.names:
                column = ops.TableColumn(table=arg, name=name)
                selections.append(translate(column))
        elif isinstance(arg, ops.Value):
            selections.append(translate(arg))
        else:
            raise com.TranslationError(
                "DataFusion backend is unable to compile selection with "
                f"operation type of {type(arg)}"
            )

    if selections:
        lf = lf.select(selections)

    if op.sort_keys:
        by = [key.name for key in op.sort_keys]
        reverse = [key.descending for key in op.sort_keys]
        lf = lf.sort(by, reverse)

    return lf


@translate.register(ops.Limit)
def limit(op):
    if op.offset:
        raise NotImplementedError("DataFusion does not support offset")
    return translate(op.table).limit(op.n)


@translate.register(ops.Aggregation)
def aggregation(op):
    lf = translate(op.table)

    if op.predicates:
        lf = lf.filter(
            functools.reduce(
                operator.and_,
                map(translate, op.predicates),
            )
        )

    if op.by:
        group_by = [translate(arg) for arg in op.by]
        lf = lf.groupby(group_by).agg
    else:
        lf = lf.select

    if op.metrics:
        metrics = [translate(arg) for arg in op.metrics]
        lf = lf(metrics)

    return lf


_join_types = {
    ops.InnerJoin: 'inner',
    ops.LeftJoin: 'left',
    ops.RightJoin: 'right',
    ops.OuterJoin: 'outer',
    ops.LeftAntiJoin: 'anti',
    ops.LeftSemiJoin: 'semi',
}


@translate.register(ops.Join)
def join(op):
    left = translate(op.left)
    right = translate(op.right)

    if isinstance(op, ops.RightJoin):
        how = 'left'
        left, right = right, left
    else:
        how = _join_types[type(op)]

    left_on, right_on = [], []
    for pred in op.predicates:
        if isinstance(pred, ops.Equals):
            left_on.append(translate(pred.left))
            right_on.append(translate(pred.right))
        else:
            raise com.TranslationError(
                "Polars backend is unable to compile join predicate "
                f"with operation type of {type(pred)}"
            )

    return left.join(right, left_on=left_on, right_on=right_on, how=how)


@translate.register(ops.DropNa)
def dropna(op):
    if op.how != 'any':
        raise com.UnsupportedArgumentError(
            f"Polars does not support how={op.how} for dropna"
        )
    if op.subset is None:
        subset = None
    elif not len(op.subset):
        return translate(op.table)
    else:
        subset = [arg.name for arg in op.subset]

    return translate(op.table).drop_nulls(subset)


@translate.register(ops.FillNa)
def fillna(op):
    table = translate(op.table)

    columns = []
    for name, dtype in op.table.schema.items():
        column = pl.col(name)
        if isinstance(op.replacements, Mapping):
            value = op.replacements.get(name)
        else:
            _assert_literal(op.replacements)
            value = op.replacements.value

        if value is not None:
            if dtype.is_floating():
                column = column.fill_nan(value)
            column = column.fill_null(value)

        # requires special treatment if the fill value has different datatype
        if dtype.is_timestamp():
            column = column.cast(pl.Datetime)

        columns.append(column)

    return table.select(columns)


@translate.register(ops.IfNull)
def ifnull(op):
    arg = translate(op.arg)
    value = translate(op.ifnull_expr)
    return arg.fill_null(value)


@translate.register(ops.ZeroIfNull)
def zeroifnull(op):
    arg = translate(op.arg)
    return arg.fill_null(0)


@translate.register(ops.NullIf)
def nullif(op):
    arg = translate(op.arg)
    null_if_expr = translate(op.null_if_expr)
    return pl.when(arg == null_if_expr).then(None).otherwise(arg)


@translate.register(ops.NullIfZero)
def nullifzero(op):
    arg = translate(op.arg)
    return pl.when(arg == 0).then(None).otherwise(arg)


@translate.register(ops.Where)
def where(op):
    bool_expr = translate(op.bool_expr)
    true_expr = translate(op.true_expr)
    false_null_expr = translate(op.false_null_expr)
    return pl.when(bool_expr).then(true_expr).otherwise(false_null_expr)


@translate.register(ops.SimpleCase)
def simple_case(op):
    base = translate(op.base)
    default = translate(op.default)
    for case, result in reversed(list(zip(op.cases, op.results))):
        case = base == translate(case)
        result = translate(result)
        default = pl.when(case).then(result).otherwise(default)
    return default


@translate.register(ops.SearchedCase)
def searched_case(op):
    default = translate(op.default)
    for case, result in reversed(list(zip(op.cases, op.results))):
        case = translate(case)
        result = translate(result)
        default = pl.when(case).then(result).otherwise(default)
    return default


@translate.register(ops.Coalesce)
def coalesce(op):
    arg = list(map(translate, op.arg))
    return pl.coalesce(arg)


@translate.register(ops.Least)
def least(op):
    arg = [translate(arg) for arg in op.arg]
    return pl.min(arg)


@translate.register(ops.Greatest)
def greatest(op):
    arg = [translate(arg) for arg in op.arg]
    return pl.max(arg)


@translate.register(ops.Contains)
def contains(op):
    value = translate(op.value)

    if isinstance(op.options, tuple):
        options = list(map(translate, op.options))
        return pl.any([value == option for option in options])
    else:
        options = translate(op.options)
        return value.is_in(options)


@translate.register(ops.NotContains)
def not_contains(op):
    value = translate(op.value)

    if isinstance(op.options, tuple):
        options = list(map(translate, op.options))
        return ~pl.any([value == option for option in options])
    else:
        options = translate(op.options)
        return ~value.is_in(options)


_string_unary = {
    ops.Strip: 'strip',
    ops.LStrip: 'lstrip',
    ops.RStrip: 'rstrip',
    ops.Lowercase: 'to_lowercase',
    ops.Uppercase: 'to_uppercase',
}


@translate.register(ops.StringLength)
def string_length(op):
    arg = translate(op.arg)
    typ = to_polars_type(op.output_dtype)
    return arg.str.lengths().cast(typ)


@translate.register(ops.StringUnary)
def string_unary(op):
    arg = translate(op.arg)
    func = _string_unary.get(type(op))
    if func is None:
        raise NotImplementedError(f'{type(op).__name__} not supported')

    method = getattr(arg.str, func)
    return method()


@translate.register(ops.Capitalize)
def captalize(op):
    arg = translate(op.arg)
    return arg.apply(lambda x: x.capitalize())


@translate.register(ops.Reverse)
def reverse(op):
    arg = translate(op.arg)
    return arg.apply(lambda x: x[::-1])


@translate.register(ops.StringSplit)
def string_split(op):
    arg = translate(op.arg)
    _assert_literal(op.delimiter)
    return arg.str.split(op.delimiter.value)


@translate.register(ops.StringReplace)
def string_replace(op):
    arg = translate(op.arg)
    pat = translate(op.pattern)
    rep = translate(op.replacement)
    return arg.str.replace(pat, rep, literal=True)


@translate.register(ops.StartsWith)
def string_startswith(op):
    arg = translate(op.arg)
    _assert_literal(op.start)
    return arg.str.starts_with(op.start.value)


@translate.register(ops.EndsWith)
def string_endswith(op):
    arg = translate(op.arg)
    _assert_literal(op.end)
    return arg.str.ends_with(op.end.value)


@translate.register(ops.StringConcat)
def string_concat(op):
    args = [translate(arg) for arg in op.arg]
    return pl.concat_str(args)


@translate.register(ops.StringJoin)
def string_join(op):
    args = [translate(arg) for arg in op.arg]
    _assert_literal(op.sep)
    return pl.concat_str(args, sep=op.sep.value)


@translate.register(ops.Substring)
def string_substrig(op):
    arg = translate(op.arg)
    _assert_literal(op.start)
    _assert_literal(op.length)
    return arg.str.slice(op.start.value, op.length.value)


@translate.register(ops.StringContains)
def string_contains(op):
    haystack = translate(op.haystack)
    _assert_literal(op.needle)
    return haystack.str.contains(op.needle.value)


@translate.register(ops.RegexSearch)
def regex_search(op):
    arg = translate(op.arg)
    _assert_literal(op.pattern)
    return arg.str.contains(op.pattern.value)


@translate.register(ops.RegexExtract)
def regex_extract(op):
    arg = translate(op.arg)
    _assert_literal(op.pattern)
    _assert_literal(op.index)
    return arg.str.extract(op.pattern.value, op.index.value)


@translate.register(ops.RegexReplace)
def regex_replace(op):
    arg = translate(op.arg)
    pattern = translate(op.pattern)
    replacement = translate(op.replacement)
    return arg.str.replace_all(pattern, replacement)


@translate.register(ops.LPad)
def lpad(op):
    arg = translate(op.arg)
    _assert_literal(op.length)
    _assert_literal(op.pad)
    return arg.str.rjust(op.length.value, op.pad.value)


@translate.register(ops.RPad)
def rpad(op):
    arg = translate(op.arg)
    _assert_literal(op.length)
    _assert_literal(op.pad)
    return arg.str.ljust(op.length.value, op.pad.value)


@translate.register(ops.StrRight)
def str_right(op):
    arg = translate(op.arg)
    _assert_literal(op.nchars)
    return arg.str.slice(-op.nchars.value, None)


@translate.register(ops.Round)
def round(op):
    arg = translate(op.arg)
    typ = to_polars_type(op.output_dtype)
    if op.digits is not None:
        _assert_literal(op.digits)
        digits = op.digits.value
    else:
        digits = 0
    return arg.round(digits).cast(typ)


@translate.register(ops.Radians)
def radians(op):
    arg = translate(op.arg)
    return arg * math.pi / 180


@translate.register(ops.Degrees)
def degrees(op):
    arg = translate(op.arg)
    return arg * 180 / math.pi


@translate.register(ops.Clip)
def clip(op):
    arg = translate(op.arg)

    if op.lower is not None and op.upper is not None:
        _assert_literal(op.lower)
        _assert_literal(op.upper)
        return arg.clip(op.lower.value, op.upper.value)
    elif op.lower is not None:
        _assert_literal(op.lower)
        return arg.clip_min(op.lower.value)
    elif op.upper is not None:
        _assert_literal(op.upper)
        return arg.clip_max(op.upper.value)
    else:
        raise com.TranslationError("No lower or upper bound specified")


@translate.register(ops.Log)
def log(op):
    arg = translate(op.arg)
    _assert_literal(op.base)
    return arg.log(op.base.value)


@translate.register(ops.Repeat)
def repeat(op):
    arg = translate(op.arg)
    _assert_literal(op.times)
    return arg.apply(lambda x: x * op.times.value)


@translate.register(ops.Sign)
def sign(op):
    arg = translate(op.arg)
    typ = to_polars_type(op.output_dtype)
    return arg.sign().cast(typ)


@translate.register(ops.Power)
def power(op):
    left = translate(op.left)
    right = translate(op.right)
    return left.pow(right)


@translate.register(ops.StructField)
def struct_field(op):
    arg = translate(op.arg)
    return arg.struct.field(op.name)


@translate.register(ops.StructColumn)
def struct_column(op):
    fields = [translate(v).alias(k) for k, v in zip(op.names, op.values)]
    return pl.struct(fields)


_reductions = {
    ops.ApproxMedian: 'median',
    ops.Count: 'count',
    ops.Max: 'max',
    ops.Mean: 'mean',
    ops.Min: 'min',
    ops.StandardDev: 'std',
    ops.Sum: 'sum',
    ops.Variance: 'var',
}


@translate.register(ops.Reduction)
def reduction(op):
    arg = translate(op.arg)
    agg = _reductions[type(op)]
    if (where := op.where) is not None:
        arg = arg.filter(translate(where))
    method = getattr(arg, agg)
    return method()


@translate.register(ops.Mode)
def mode(op):
    arg = translate(op.arg)
    if (where := op.where) is not None:
        arg = arg.filter(translate(where))
    return arg.mode().min()


@translate.register(ops.Distinct)
def distinct(op):
    table = translate(op.table)
    return table.unique()


@translate.register(ops.CountStar)
def count_star(op):
    if (where := op.where) is not None:
        condition = translate(where)
        return condition.filter(condition).count()
    return pl.count()


@translate.register(ops.TimestampNow)
def timestamp_now(op):
    now = pd.Timestamp("now", tz="UTC").tz_localize(None)
    return pl.lit(now)


@translate.register(ops.Strftime)
def strftime(op):
    arg = translate(op.arg)
    _assert_literal(op.format_str)
    return arg.dt.strftime(op.format_str.value)


@translate.register(ops.Date)
def date(op):
    arg = translate(op.arg)
    return arg.cast(pl.Date)


@translate.register(ops.DateTruncate)
@translate.register(ops.TimestampTruncate)
def temporal_truncate(op):
    arg = translate(op.arg)
    unit = "mo" if op.unit == "M" else op.unit
    unit = f"1{unit.lower()}"
    return arg.dt.truncate(unit, "-1w")


@translate.register(ops.DateFromYMD)
def date_from_ymd(op):
    return pl.date(
        year=translate(op.year),
        month=translate(op.month),
        day=translate(op.day),
    )


@translate.register(ops.Atan2)
def atan2(op):
    left = translate(op.left)
    right = translate(op.right)
    return pl.map([left, right], lambda cols: np.arctan2(cols[0], cols[1]))


@translate.register(ops.Modulus)
def modulus(op):
    left = translate(op.left)
    right = translate(op.right)
    return pl.map([left, right], lambda cols: np.mod(cols[0], cols[1]))


@translate.register(ops.TimestampFromYMDHMS)
def timestamp_from_ymdhms(op):
    return pl.datetime(
        year=translate(op.year),
        month=translate(op.month),
        day=translate(op.day),
        hour=translate(op.hours),
        minute=translate(op.minutes),
        second=translate(op.seconds),
    )


@translate.register(ops.TimestampFromUNIX)
def timestamp_from_unix(op):
    arg = translate(op.arg)
    unit = op.unit
    if unit == "s":
        arg = arg.cast(pl.Int64) * 1_000
        unit = "ms"
    return arg.cast(pl.Datetime).dt.with_time_unit(unit)


@translate.register(ops.IntervalFromInteger)
def interval_from_integer(op):
    arg = translate(op.arg)
    return _make_duration(arg, dt.Interval(unit=op.unit))


@translate.register(ops.StringToTimestamp)
def string_to_timestamp(op):
    arg = translate(op.arg)
    _assert_literal(op.format_str)
    return arg.str.strptime(pl.Datetime, op.format_str.value)


@translate.register(ops.TimestampAdd)
def timestamp_add(op):
    left = translate(op.left)
    right = translate(op.right)
    return left + right


@translate.register(ops.TimestampSub)
@translate.register(ops.DateDiff)
@translate.register(ops.IntervalSubtract)
def timestamp_sub(op):
    left = translate(op.left)
    right = translate(op.right)
    return left - right


@translate.register(ops.TimestampDiff)
def timestamp_diff(op):
    left = translate(op.left)
    right = translate(op.right)
    # TODO: truncating both to seconds is necessary to conform to the output
    # type of the operation
    return left.dt.truncate("1s") - right.dt.truncate("1s")


@translate.register(ops.ArrayLength)
def array_length(op):
    arg = translate(op.arg)
    return arg.arr.lengths()


@translate.register(ops.ArrayConcat)
def array_concat(op):
    left = translate(op.left)
    right = translate(op.right)
    return left.arr.concat(right)


@translate.register(ops.ArrayColumn)
def array_column(op):
    cols = list(map(translate, op.cols))
    return pl.concat_list(cols)


@translate.register(ops.ArrayCollect)
def array_collect(op):
    arg = translate(op.arg)
    if (where := op.where) is not None:
        arg = arg.filter(translate(where))
    return arg.list()


@translate.register(ops.Unnest)
def unnest(op):
    arg = translate(op.arg)
    return arg.explode()


_date_methods = {
    ops.ExtractDay: "day",
    ops.ExtractMonth: "month",
    ops.ExtractYear: "year",
    ops.ExtractQuarter: "quarter",
    ops.ExtractDayOfYear: "ordinal_day",
    ops.ExtractWeekOfYear: "week",
    ops.ExtractHour: "hour",
    ops.ExtractMinute: "minute",
    ops.ExtractSecond: "second",
    ops.ExtractMillisecond: "millisecond",
}


@translate.register(ops.ExtractTemporalField)
def extract_date_field(op):
    arg = translate(op.arg)
    method = operator.methodcaller(_date_methods[type(op)])
    return method(arg.dt).cast(pl.Int32)


@translate.register(ops.ExtractEpochSeconds)
def extract_epoch_seconds(op):
    arg = translate(op.arg)
    return arg.dt.epoch('s').cast(pl.Int32)


_day_of_week_offset = vparse(pl.__version__) >= vparse("0.15.1")


_unary = {
    # TODO(kszucs): factor out the lambdas
    ops.Abs: operator.methodcaller('abs'),
    ops.Acos: operator.methodcaller('arccos'),
    ops.Asin: operator.methodcaller('arcsin'),
    ops.Atan: operator.methodcaller('arctan'),
    ops.Ceil: lambda arg: arg.ceil().cast(pl.Int64),
    ops.Cos: operator.methodcaller('cos'),
    ops.Cot: lambda arg: 1.0 / arg.tan(),
    ops.DayOfWeekIndex: (
        lambda arg: arg.dt.weekday().cast(pl.Int16) - _day_of_week_offset
    ),
    ops.Exp: operator.methodcaller('exp'),
    ops.Floor: lambda arg: arg.floor().cast(pl.Int64),
    ops.IsInf: operator.methodcaller('is_infinite'),
    ops.IsNan: operator.methodcaller('is_nan'),
    ops.IsNull: operator.methodcaller('is_null'),
    ops.Ln: operator.methodcaller('log'),
    ops.Log10: operator.methodcaller('log10'),
    ops.Log2: lambda arg: arg.log(2),
    ops.Negate: operator.neg,
    ops.Not: operator.methodcaller('is_not'),
    ops.NotNull: operator.methodcaller('is_not_null'),
    ops.Sin: operator.methodcaller('sin'),
    ops.Sqrt: operator.methodcaller('sqrt'),
    ops.Tan: operator.methodcaller('tan'),
}


@translate.register(ops.DayOfWeekName)
def day_of_week_name(op):
    index = translate(op.arg).dt.weekday() - _day_of_week_offset
    arg = None
    for i, name in enumerate(calendar.day_name):
        arg = pl.when(index == i).then(name).otherwise(arg)
    return arg


@translate.register(ops.Unary)
def unary(op):
    arg = translate(op.arg)
    func = _unary.get(type(op))
    if func is None:
        raise NotImplementedError(f'{type(op).__name__} not supported')
    return func(arg)


_comparisons = {
    ops.Equals: operator.eq,
    ops.Greater: operator.gt,
    ops.GreaterEqual: operator.ge,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.NotEquals: operator.ne,
}


@translate.register(ops.Comparison)
def comparison(op):
    left = translate(op.left)
    right = translate(op.right)
    func = _comparisons.get(type(op))
    if func is None:
        raise NotImplementedError(f'{type(op).__name__} not supported')
    return func(left, right)


@translate.register(ops.Between)
def between(op):
    arg = translate(op.arg)
    lower = translate(op.lower_bound)
    upper = translate(op.upper_bound)
    return arg.is_between(lower, upper, closed="both")


_bitwise_binops = {
    ops.BitwiseRightShift: np.right_shift,
    ops.BitwiseLeftShift: np.left_shift,
    ops.BitwiseOr: np.bitwise_or,
    ops.BitwiseAnd: np.bitwise_and,
    ops.BitwiseXor: np.bitwise_xor,
}


@translate.register(ops.BitwiseBinary)
def bitwise_binops(op):
    ufunc = _bitwise_binops.get(type(op))
    if ufunc is None:
        raise NotImplementedError(f'{type(op).__name__} not supported')
    left = translate(op.left)
    right = translate(op.right)

    if isinstance(op.right, ops.Literal):
        result = left.map(lambda col: ufunc(col, op.right.value))
    elif isinstance(op.left, ops.Literal):
        result = right.map(lambda col: ufunc(op.left.value, col))
    else:
        result = pl.map([left, right], lambda cols: ufunc(cols[0], cols[1]))

    return result.cast(to_polars_type(op.output_dtype))


@translate.register(ops.BitwiseNot)
def bitwise_not(op):
    arg = translate(op.arg)
    return arg.map(lambda x: np.invert(x))


_binops = {
    ops.Add: operator.add,
    ops.And: operator.and_,
    ops.DateAdd: operator.add,
    ops.DateSub: operator.sub,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Multiply: operator.mul,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,
    ops.Subtract: operator.sub,
}


@translate.register(ops.Binary)
def binop(op):
    left = translate(op.left)
    right = translate(op.right)
    func = _binops.get(type(op))
    if func is None:
        raise NotImplementedError(f'{type(op).__name__} not supported')
    return func(left, right)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op):
    func_args = list(map(translate, op.func_args))
    return_type = to_polars_type(op.return_type)

    return pl.map(func_args, lambda args: op.func(*args), return_dtype=return_type)
