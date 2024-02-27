from __future__ import annotations

import calendar
import math
import operator
from collections.abc import Mapping
from functools import partial, reduce, singledispatch
from math import isnan

import numpy as np
import pandas as pd
import polars as pl
from packaging.version import parse as vparse

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.pandas.rewrites import PandasAsofJoin, PandasJoin, PandasRename
from ibis.expr.operations.udf import InputType
from ibis.formats.polars import PolarsType
from ibis.util import gen_name


def _expr_method(expr, op, methods):
    for m in methods:
        if hasattr(expr, m):
            return getattr(expr, m)
    raise com.TranslationError(
        f"Failed to translate {op}; expected expression method(s) not found:\n{methods!r}"
    )


def _literal_value(op, nan_as_none=False):
    # TODO(kszucs): broadcast and apply UDF on two columns using concat_list
    # TODO(kszucs): better error message
    if op is None:
        return None
    elif not isinstance(op, ops.Literal):
        raise com.UnsupportedArgumentError(
            f"Polars does not support columnar argument {op.name}"
        )
    else:
        value = op.value
        return None if nan_as_none and isnan(value) else value


@singledispatch
def translate(expr, *, ctx):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op, **_):
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate.register(ops.DatabaseTable)
def table(op, **_):
    return op.source._tables[op.name]


@translate.register(ops.DummyTable)
def dummy_table(op, **kw):
    selections = [translate(arg, **kw) for name, arg in op.values.items()]
    return pl.DataFrame().lazy().select(selections)


@translate.register(ops.InMemoryTable)
def in_memory_table(op, **_):
    return op.data.to_polars(op.schema).lazy()


@translate.register(ops.Alias)
def alias(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.alias(op.name)


def _make_duration(value, dtype):
    kwargs = {f"{dtype.resolution}s": value}
    return pl.duration(**kwargs)


@translate.register(ops.Literal)
def literal(op, **_):
    value = op.value
    dtype = op.dtype

    if dtype.is_array():
        value = pl.Series("", value)
        typ = PolarsType.from_ibis(dtype)
        val = pl.lit(value, dtype=typ)
        return val.implode()
    elif dtype.is_struct():
        values = [
            pl.lit(v, dtype=PolarsType.from_ibis(dtype[k])).alias(k)
            for k, v in value.items()
        ]
        return pl.struct(values)
    elif dtype.is_interval():
        return _make_duration(value, dtype)
    elif dtype.is_null():
        return pl.lit(value)
    elif dtype.is_binary():
        return pl.lit(value)
    else:
        typ = PolarsType.from_ibis(dtype)
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
def cast(op, **_):
    return _cast(op, strict=True)


@translate.register(ops.TryCast)
def try_cast(op, **_):
    return _cast(op, strict=False)


def _cast(op, strict=True, **kw):
    arg = translate(op.arg, **kw)
    dtype = op.arg.dtype
    to = op.to

    if to.is_interval():
        if not strict:
            raise NotImplementedError(f"Unsupported try_cast to type: {to!r}")
        return _make_duration(arg, to)
    elif to.is_date():
        if not strict:
            raise NotImplementedError(f"Unsupported try_cast to type: {to!r}")
        if dtype.is_string():
            return arg.str.strptime(pl.Date, "%Y-%m-%d")
    elif to.is_timestamp():
        if not strict:
            raise NotImplementedError(f"Unsupported try_cast to type: {to!r}")

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

    typ = PolarsType.from_ibis(to)
    return arg.cast(typ, strict=strict)


@translate.register(ops.Field)
def column(op, **_):
    return pl.col(op.name)


@translate.register(ops.SortKey)
def sort_key(op, **kw):
    arg = translate(op.expr, **kw)
    descending = op.descending
    try:
        return arg.sort(descending=descending)
    except TypeError:  # pragma: no cover
        return arg.sort(reverse=descending)  # pragma: no cover


@translate.register(ops.Project)
def project(op, **kw):
    lf = translate(op.parent, **kw)

    selections = []
    unnests = []
    for name, arg in op.values.items():
        if isinstance(arg, ops.Unnest):
            unnests.append(name)
            translated = translate(arg.arg, **kw)
        elif isinstance(arg, ops.Value):
            translated = translate(arg, **kw)
        else:
            raise com.TranslationError(
                "Polars backend is unable to compile selection with "
                f"operation type of {type(arg)}"
            )
        selections.append(translated.alias(name))

    if selections:
        lf = lf.select(selections)

        if unnests:
            lf = lf.explode(*unnests)

    return lf


@translate.register(ops.Sort)
def sort(op, **kw):
    lf = translate(op.parent, **kw)

    if op.keys:
        by = [key.name for key in op.keys]
        descending = [key.descending for key in op.keys]
        try:
            lf = lf.sort(by, descending=descending, nulls_last=True)
        except TypeError:  # pragma: no cover
            lf = lf.sort(by, reverse=descending, nulls_last=True)  # pragma: no cover

    return lf


@translate.register(ops.Filter)
def filter_(op, **kw):
    lf = translate(op.parent, **kw)

    if op.predicates:
        predicates = map(partial(translate, **kw), op.predicates)
        predicate = reduce(operator.and_, predicates)
        lf = lf.filter(predicate)

    return lf


@translate.register(ops.Limit)
def limit(op, **kw):
    if (n := op.n) is not None and not isinstance(n, int):
        raise NotImplementedError("Dynamic limit not supported")

    if not isinstance(offset := op.offset, int):
        raise NotImplementedError("Dynamic offset not supported")

    lf = translate(op.parent, **kw)
    return lf.slice(offset, n)


@translate.register(ops.Aggregate)
def aggregation(op, **kw):
    lf = translate(op.parent, **kw)

    if op.groups:
        # project first to handle computed group by columns
        lf = (
            lf.with_columns(
                [translate(arg, **kw).alias(name) for name, arg in op.groups.items()]
            )
            .group_by(list(op.groups.keys()))
            .agg
        )
    else:
        lf = lf.select

    if op.metrics:
        metrics = [translate(arg, **kw).alias(name) for name, arg in op.metrics.items()]
        lf = lf(metrics)

    return lf


@translate.register(PandasRename)
def rename(op, **kw):
    parent = translate(op.parent, **kw)
    return parent.rename(op.mapping)


@translate.register(PandasJoin)
def join(op, **kw):
    how = op.how
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)

    # workaround required for https://github.com/pola-rs/polars/issues/13130
    prefix = gen_name("on")
    left_on = {f"{prefix}_{i}": translate(v, **kw) for i, v in enumerate(op.left_on)}
    right_on = {f"{prefix}_{i}": translate(v, **kw) for i, v in enumerate(op.right_on)}
    left = left.with_columns(**left_on)
    right = right.with_columns(**right_on)
    on = list(left_on.keys())

    if how == "right":
        how = "left"
        left, right = right, left

    joined = left.join(right, on=on, how=how)

    try:
        joined = joined.drop(*on)
    except TypeError:
        joined = joined.drop(columns=on)

    return joined


@translate.register(PandasAsofJoin)
def asof_join(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)

    # workaround required for https://github.com/pola-rs/polars/issues/13130
    on, by = gen_name("on"), gen_name("by")
    left_on = {f"{on}_{i}": translate(v, **kw) for i, v in enumerate(op.left_on)}
    right_on = {f"{on}_{i}": translate(v, **kw) for i, v in enumerate(op.right_on)}
    left_by = {f"{by}_{i}": translate(v, **kw) for i, v in enumerate(op.left_by)}
    right_by = {f"{by}_{i}": translate(v, **kw) for i, v in enumerate(op.right_by)}

    left = left.with_columns(**left_on, **left_by)
    right = right.with_columns(**right_on, **right_by)

    on = list(left_on.keys())
    by = list(left_by.keys())

    if op.operator in {ops.Less, ops.LessEqual}:
        direction = "forward"
    elif op.operator in {ops.Greater, ops.GreaterEqual}:
        direction = "backward"
    elif op.operator == ops.Equals:
        direction = "nearest"
    else:
        raise NotImplementedError(f"Operator {operator} not supported for asof join")

    assert len(on) == 1
    joined = left.join_asof(right, on=on[0], by=by, strategy=direction)
    try:
        joined = joined.drop(*(on + by))
    except TypeError:
        joined = joined.drop(columns=on + by)
    return joined


@translate.register(ops.DropNa)
def dropna(op, **kw):
    lf = translate(op.parent, **kw)

    if op.subset is None:
        subset = None
    elif not len(op.subset):
        return lf.clear() if op.how == "all" else lf
    else:
        subset = [arg.name for arg in op.subset]

    if op.how == "all":
        cols = pl.col(subset) if subset else pl.all()
        return lf.filter(~pl.all_horizontal(cols.is_null()))

    return lf.drop_nulls(subset)


@translate.register(ops.FillNa)
def fillna(op, **kw):
    table = translate(op.parent, **kw)

    columns = []

    repls = op.replacements

    if isinstance(repls, Mapping):

        def get_replacement(name):
            repl = repls.get(name)
            if repl is not None:
                return repl.value
            else:
                return None

    else:
        value = repls.value

        def get_replacement(_):
            return value

    for name, dtype in op.parent.schema.items():
        column = pl.col(name)
        if isinstance(op.replacements, Mapping):
            value = op.replacements.get(name)
        else:
            value = _literal_value(op.replacements)

        if value is not None:
            if dtype.is_floating():
                column = column.fill_nan(value)
            column = column.fill_null(value)

        # requires special treatment if the fill value has different datatype
        if dtype.is_timestamp():
            column = column.cast(pl.Datetime)

        columns.append(column)

    return table.select(columns)


@translate.register(ops.IdenticalTo)
def identical_to(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.eq_missing(right)


@translate.register(ops.NullIf)
def nullif(op, **kw):
    arg = translate(op.arg, **kw)
    null_if_expr = translate(op.null_if_expr, **kw)
    return pl.when(arg == null_if_expr).then(None).otherwise(arg)


@translate.register(ops.IfElse)
def ifelse(op, **kw):
    bool_expr = translate(op.bool_expr, **kw)
    true_expr = translate(op.true_expr, **kw)
    false_null_expr = translate(op.false_null_expr, **kw)
    return pl.when(bool_expr).then(true_expr).otherwise(false_null_expr)


@translate.register(ops.SimpleCase)
def simple_case(op, **kw):
    base = translate(op.base, **kw)
    default = translate(op.default, **kw)
    for case, result in reversed(list(zip(op.cases, op.results))):
        case = base == translate(case, **kw)
        result = translate(result, **kw)
        default = pl.when(case).then(result).otherwise(default)
    return default


@translate.register(ops.SearchedCase)
def searched_case(op, **kw):
    default = translate(op.default, **kw)
    for case, result in reversed(list(zip(op.cases, op.results))):
        case = translate(case, **kw)
        result = translate(result, **kw)
        default = pl.when(case).then(result).otherwise(default)
    return default


@translate.register(ops.Coalesce)
def coalesce(op, **kw):
    arg = [translate(expr, **kw) for expr in op.arg]
    return pl.coalesce(arg)


@translate.register(ops.Least)
def least(op, **kw):
    arg = [translate(arg, **kw) for arg in op.arg]
    return pl.min_horizontal(arg)


@translate.register(ops.Greatest)
def greatest(op, **kw):
    arg = [translate(arg, **kw) for arg in op.arg]
    return pl.max_horizontal(arg)


@translate.register(ops.InSubquery)
def in_column(op, **kw):
    value = translate(op.value, **kw)
    needle = translate(op.needle, **kw)
    return needle.is_in(value)


@translate.register(ops.InValues)
def in_values(op, **kw):
    value = translate(op.value, **kw)
    options = list(map(translate, op.options))
    if not options:
        return pl.lit(False)
    return pl.any_horizontal([value == option for option in options])


_string_unary = {
    ops.Strip: "strip_chars",
    ops.LStrip: "strip_chars_start",
    ops.RStrip: "strip_chars_end",
    ops.Lowercase: "to_lowercase",
    ops.Uppercase: "to_uppercase",
}


@translate.register(ops.StringLength)
def string_length(op, **kw):
    arg = translate(op.arg, **kw)
    typ = PolarsType.from_ibis(op.dtype)
    return arg.str.len_bytes().cast(typ)


@translate.register(ops.Capitalize)
def capitalize(op, **kw):
    arg = translate(op.arg, **kw)
    typ = PolarsType.from_ibis(op.dtype)
    first = arg.str.slice(0, 1).str.to_uppercase()
    rest = arg.str.slice(1, None).str.to_lowercase()
    return (first + rest).cast(typ)


@translate.register(ops.StringUnary)
def string_unary(op, **kw):
    arg = translate(op.arg, **kw)
    func = _string_unary.get(type(op))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")

    method = getattr(arg.str, func)
    return method()


@translate.register(ops.Reverse)
def reverse(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.map_elements(lambda x: x[::-1])


@translate.register(ops.StringSplit)
def string_split(op, **kw):
    arg = translate(op.arg, **kw)
    delim = _literal_value(op.delimiter)
    return arg.str.split(by=delim)


@translate.register(ops.StringReplace)
def string_replace(op, **kw):
    arg = translate(op.arg, **kw)
    pat = translate(op.pattern, **kw)
    rep = translate(op.replacement, **kw)
    return arg.str.replace(pat, rep, literal=True)


@translate.register(ops.StartsWith)
def string_startswith(op, **kw):
    arg = translate(op.arg, **kw)
    start = _literal_value(op.start)
    return arg.str.starts_with(start)


@translate.register(ops.EndsWith)
def string_endswith(op, **kw):
    arg = translate(op.arg, **kw)
    end = _literal_value(op.end)
    return arg.str.ends_with(end)


@translate.register(ops.StringConcat)
def string_concat(op, **kw):
    args = [translate(arg, **kw) for arg in op.arg]
    return pl.concat_str(args)


@translate.register(ops.StringJoin)
def string_join(op, **kw):
    args = [translate(arg, **kw) for arg in op.arg]
    sep = _literal_value(op.sep)
    return pl.concat_str(args, separator=sep)


@translate.register(ops.Substring)
def string_substring(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.slice(
        offset=_literal_value(op.start),
        length=_literal_value(op.length),
    )


@translate.register(ops.StringContains)
def string_contains(op, **kw):
    haystack = translate(op.haystack, **kw)
    return haystack.str.contains(
        pattern=_literal_value(op.needle),
        literal=True,
    )


@translate.register(ops.RegexSearch)
def regex_search(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.contains(
        pattern=_literal_value(op.pattern),
        literal=False,
    )


@translate.register(ops.RegexExtract)
def regex_extract(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.extract(
        pattern=_literal_value(op.pattern),
        group_index=_literal_value(op.index),
    )


@translate.register(ops.RegexReplace)
def regex_replace(op, **kw):
    arg = translate(op.arg, **kw)
    pattern = translate(op.pattern, **kw)
    replacement = translate(op.replacement, **kw)
    return arg.str.replace_all(
        pattern=pattern,
        value=replacement,
    )


@translate.register(ops.LPad)
def lpad(op, **kw):
    arg = translate(op.arg, **kw)
    _lpad = _expr_method(arg.str, "lpad", ["pad_start", "rjust"])
    return _lpad(_literal_value(op.length), _literal_value(op.pad))


@translate.register(ops.RPad)
def rpad(op, **kw):
    arg = translate(op.arg, **kw)
    _rpad = _expr_method(arg.str, "rpad", ["pad_end", "ljust"])
    return _rpad(_literal_value(op.length), _literal_value(op.pad))


@translate.register(ops.StrRight)
def str_right(op, **kw):
    arg = translate(op.arg, **kw)
    nchars = _literal_value(op.nchars)
    return arg.str.slice(-nchars, None)


@translate.register(ops.Round)
def round(op, **kw):
    arg = translate(op.arg, **kw)
    typ = PolarsType.from_ibis(op.dtype)
    digits = _literal_value(op.digits)
    return arg.round(digits or 0).cast(typ)


@translate.register(ops.Radians)
def radians(op, **kw):
    arg = translate(op.arg, **kw)
    return arg * math.pi / 180


@translate.register(ops.Degrees)
def degrees(op, **kw):
    arg = translate(op.arg, **kw)
    return arg * 180 / math.pi


@translate.register(ops.Clip)
def clip(op, **kw):
    arg = translate(op.arg, **kw)

    lower = _literal_value(op.lower)
    upper = _literal_value(op.upper)

    if vparse(pl.__version__) >= vparse("0.19.12"):
        if not (lower is None and upper is None):
            return arg.clip(lower, upper)
    elif lower is not None and upper is not None:
        return arg.clip(lower, upper)
    elif lower is not None:
        return arg.clip_min(lower)
    elif upper is not None:
        return arg.clip_max(upper)

    raise com.TranslationError("No lower or upper bound specified")


@translate.register(ops.Log)
def log(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.log(base=_literal_value(op.base))


@translate.register(ops.Repeat)
def repeat(op, **kw):
    arg = translate(op.arg, **kw)
    n_times = _literal_value(op.times)
    return pl.concat_str([arg] * n_times, separator="")


@translate.register(ops.Sign)
def sign(op, **kw):
    arg = translate(op.arg, **kw)
    typ = PolarsType.from_ibis(op.dtype)
    return arg.sign().cast(typ)


@translate.register(ops.Power)
def power(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.pow(right)


@translate.register(ops.StructField)
def struct_field(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.struct.field(op.name)


@translate.register(ops.StructColumn)
def struct_column(op, **kw):
    fields = [translate(v, **kw).alias(k) for k, v in zip(op.names, op.values)]
    return pl.struct(fields)


_reductions = {
    ops.All: "all",
    ops.Any: "any",
    ops.ApproxMedian: "median",
    ops.Count: "count",
    ops.CountDistinct: "n_unique",
    ops.First: "first",
    ops.Last: "last",
    ops.Max: "max",
    ops.Mean: "mean",
    ops.Median: "median",
    ops.Min: "min",
    ops.Mode: "mode",
    ops.StandardDev: "std",
    ops.Sum: "sum",
    ops.Variance: "var",
}

for reduction in _reductions.keys():

    @translate.register(reduction)
    def reduction(op, **kw):
        args = [
            translate(arg, **kw)
            for name, arg in zip(op.argnames, op.args)
            if name not in ("where", "how")
        ]

        agg = _reductions[type(op)]

        predicates = [arg.is_not_null() for arg in args]
        if (where := op.where) is not None:
            predicates.append(translate(where, **kw))

        first, *rest = args
        method = operator.methodcaller(agg, *rest)
        return method(first.filter(reduce(operator.and_, predicates))).cast(
            PolarsType.from_ibis(op.dtype)
        )


@translate.register(ops.Quantile)
def execute_quantile(op, **kw):
    arg = translate(op.arg, **kw)
    quantile = translate(op.quantile, **kw)
    filt = arg.is_not_null() & quantile.is_not_null()
    if (where := op.where) is not None:
        filt &= translate(where, **kw)

    # we can't throw quantile into the _reductions mapping because Polars'
    # default interpolation of "nearest" doesn't match the rest of our backends
    return arg.filter(filt).quantile(quantile, interpolation="linear")


@translate.register(ops.Correlation)
def correlation(op, **kw):
    x = op.left
    if (x_type := x.dtype).is_boolean():
        x = ops.Cast(x, dt.Int32(nullable=x_type.nullable))

    y = op.right
    if (y_type := y.dtype).is_boolean():
        y = ops.Cast(y, dt.Int32(nullable=y_type.nullable))

    if (where := op.where) is not None:
        x = ops.IfElse(where, x, None)
        y = ops.IfElse(where, y, None)

    return pl.corr(translate(x, **kw), translate(y, **kw))


@translate.register(ops.Distinct)
def distinct(op, **kw):
    table = translate(op.parent, **kw)
    return table.unique()


@translate.register(ops.CountStar)
def count_star(op, **kw):
    if (where := op.where) is not None:
        condition = translate(where, **kw)
        result = condition.sum()
    else:
        try:
            result = pl.len()
        except AttributeError:
            result = pl.count()
    return result.cast(PolarsType.from_ibis(op.dtype))


@translate.register(ops.TimestampNow)
def timestamp_now(op, **_):
    return pl.lit(pd.Timestamp("now", tz="UTC").tz_localize(None))


@translate.register(ops.Strftime)
def strftime(op, **kw):
    arg = translate(op.arg, **kw)
    fmt = _literal_value(op.format_str)
    return arg.dt.strftime(format=fmt)


@translate.register(ops.Date)
def date(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.cast(pl.Date)


@translate.register(ops.DateTruncate)
@translate.register(ops.TimestampTruncate)
def temporal_truncate(op, **kw):
    arg = translate(op.arg, **kw)
    unit = "mo" if op.unit.short == "M" else op.unit.short
    unit = f"1{unit.lower()}"
    return arg.dt.truncate(unit, "-1w")


def _compile_literal_interval(op):
    if not isinstance(op, ops.Literal):
        raise com.UnsupportedOperationError(
            "Only literal interval values are supported"
        )

    if op.dtype.unit.short == "M":
        suffix = "mo"
    else:
        suffix = op.dtype.unit.short.lower()

    return f"{op.value}{suffix}"


@translate.register(ops.TimestampBucket)
def timestamp_bucket(op, **kw):
    arg = translate(op.arg, **kw)
    interval = _compile_literal_interval(op.interval)
    if op.offset is not None:
        offset = _compile_literal_interval(op.offset)
        neg_offset = offset[1:] if offset.startswith("-") else f"-{offset}"
        arg = arg.dt.offset_by(neg_offset)
    else:
        offset = None
    res = arg.dt.truncate(interval, offset)
    return res


@translate.register(ops.DateFromYMD)
def date_from_ymd(op, **kw):
    return pl.date(
        year=translate(op.year, **kw),
        month=translate(op.month, **kw),
        day=translate(op.day, **kw),
    )


@translate.register(ops.Atan2)
def atan2(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return pl.map_batches([left, right], lambda cols: np.arctan2(cols[0], cols[1]))


@translate.register(ops.Modulus)
def modulus(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return pl.map_batches([left, right], lambda cols: np.mod(cols[0], cols[1]))


@translate.register(ops.TimestampFromYMDHMS)
def timestamp_from_ymdhms(op, **kw):
    return pl.datetime(
        year=translate(op.year, **kw),
        month=translate(op.month, **kw),
        day=translate(op.day, **kw),
        hour=translate(op.hours, **kw),
        minute=translate(op.minutes, **kw),
        second=translate(op.seconds, **kw),
    )


@translate.register(ops.TimestampFromUNIX)
def timestamp_from_unix(op, **kw):
    arg = translate(op.arg, **kw)
    unit = op.unit.short
    if unit == "s":
        arg = arg.cast(pl.Int64) * 1_000
        unit = "ms"
    return arg.cast(pl.Datetime(time_unit=unit))


@translate.register(ops.IntervalFromInteger)
def interval_from_integer(op, **kw):
    arg = translate(op.arg, **kw)
    return _make_duration(arg, dt.Interval(unit=op.unit))


@translate.register(ops.StringToTimestamp)
def string_to_timestamp(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.strptime(
        dtype=pl.Datetime,
        format=_literal_value(op.format_str),
    )


@translate.register(ops.TimestampDiff)
def timestamp_diff(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    # TODO: truncating both to seconds is necessary to conform to the output
    # type of the operation
    return left.dt.truncate("1s") - right.dt.truncate("1s")


@translate.register(ops.ArrayLength)
def array_length(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.list.len()


@translate.register(ops.ArrayConcat)
def array_concat(op, **kw):
    result, *rest = map(partial(translate, **kw), op.arg)

    for arg in rest:
        result = result.list.concat(arg)

    return result


@translate.register(ops.Array)
def array_column(op, **kw):
    cols = [translate(col, **kw) for col in op.exprs]
    return pl.concat_list(cols)


@translate.register(ops.ArrayCollect)
def array_collect(op, **kw):
    arg = translate(op.arg, **kw)
    if (where := op.where) is not None:
        arg = arg.filter(translate(where, **kw))
    return arg


@translate.register(ops.ArrayFlatten)
def array_flatten(op, **kw):
    return pl.concat_list(translate(op.arg, **kw))


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
    ops.ExtractMicrosecond: "microsecond",
    ops.ExtractMillisecond: "millisecond",
}


@translate.register(ops.ExtractTemporalField)
def extract_date_field(op, **kw):
    arg = translate(op.arg, **kw)
    method = operator.methodcaller(_date_methods[type(op)])
    return method(arg.dt).cast(pl.Int32)


@translate.register(ops.ExtractEpochSeconds)
def extract_epoch_seconds(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.dt.epoch("s").cast(pl.Int32)


_day_of_week_offset = vparse(pl.__version__) >= vparse("0.15.1")


_unary = {
    # TODO(kszucs): factor out the lambdas
    ops.Abs: operator.methodcaller("abs"),
    ops.Acos: operator.methodcaller("arccos"),
    ops.Asin: operator.methodcaller("arcsin"),
    ops.Atan: operator.methodcaller("arctan"),
    ops.Ceil: lambda arg: arg.ceil().cast(pl.Int64),
    ops.Cos: operator.methodcaller("cos"),
    ops.Cot: lambda arg: 1.0 / arg.tan(),
    ops.DayOfWeekIndex: (
        lambda arg: arg.dt.weekday().cast(pl.Int16) - _day_of_week_offset
    ),
    ops.Exp: operator.methodcaller("exp"),
    ops.Floor: lambda arg: arg.floor().cast(pl.Int64),
    ops.IsInf: operator.methodcaller("is_infinite"),
    ops.IsNan: operator.methodcaller("is_nan"),
    ops.IsNull: operator.methodcaller("is_null"),
    ops.Ln: operator.methodcaller("log"),
    ops.Log10: operator.methodcaller("log10"),
    ops.Log2: lambda arg: arg.log(2),
    ops.Negate: operator.neg,
    ops.Not: operator.methodcaller("not_"),
    ops.NotNull: operator.methodcaller("is_not_null"),
    ops.Sin: operator.methodcaller("sin"),
    ops.Sqrt: operator.methodcaller("sqrt"),
    ops.Tan: operator.methodcaller("tan"),
}


@translate.register(ops.DayOfWeekName)
def day_of_week_name(op, **kw):
    index = translate(op.arg, **kw).dt.weekday() - _day_of_week_offset
    arg = None
    for i, name in enumerate(calendar.day_name):
        arg = pl.when(index == i).then(pl.lit(name)).otherwise(arg)
    return arg


@translate.register(ops.Unary)
def unary(op, **kw):
    arg = translate(op.arg, **kw)
    func = _unary.get(type(op))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
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
def comparison(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    func = _comparisons.get(type(op))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
    return func(left, right)


@translate.register(ops.Between)
def between(op, **kw):
    op_arg = op.arg
    arg = translate(op_arg, **kw)
    dtype = op_arg.dtype
    lower = translate(ops.Cast(op.lower_bound, dtype), **kw)
    upper = translate(ops.Cast(op.upper_bound, dtype), **kw)
    return arg.is_between(lower, upper, closed="both")


_bitwise_binops = {
    ops.BitwiseRightShift: np.right_shift,
    ops.BitwiseLeftShift: np.left_shift,
    ops.BitwiseOr: np.bitwise_or,
    ops.BitwiseAnd: np.bitwise_and,
    ops.BitwiseXor: np.bitwise_xor,
}


@translate.register(ops.BitwiseBinary)
def bitwise_binops(op, **kw):
    ufunc = _bitwise_binops.get(type(op))
    if ufunc is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)

    if isinstance(op.right, ops.Literal):
        result = left.map_batches(lambda col: ufunc(col, op.right.value))
    elif isinstance(op.left, ops.Literal):
        result = right.map_batches(lambda col: ufunc(op.left.value, col))
    else:
        result = pl.map_batches([left, right], lambda cols: ufunc(cols[0], cols[1]))

    return result.cast(PolarsType.from_ibis(op.dtype))


@translate.register(ops.BitwiseNot)
def bitwise_not(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.map_batches(lambda x: np.invert(x))


_binops = {
    ops.Add: operator.add,
    ops.And: operator.and_,
    ops.DateAdd: operator.add,
    ops.DateSub: operator.sub,
    ops.DateDiff: operator.sub,
    ops.TimestampAdd: operator.add,
    ops.TimestampSub: operator.sub,
    ops.IntervalSubtract: operator.sub,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Multiply: operator.mul,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,
    ops.Subtract: operator.sub,
}


@translate.register(ops.Binary)
def binop(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    func = _binops.get(type(op))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
    return func(left, right)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op, **kw):
    func_args = [translate(arg, **kw) for arg in op.func_args]
    return_type = PolarsType.from_ibis(op.return_type)

    return pl.map_batches(
        func_args, lambda args: op.func(*args), return_dtype=return_type
    )


@translate.register(ops.E)
def execute_e(op, **_):
    return pl.lit(np.e)


@translate.register(ops.Pi)
def execute_pi(op, **_):
    return pl.lit(np.pi)


@translate.register(ops.Time)
def execute_time(op, **kw):
    arg = translate(op.arg, **kw)
    if op.arg.dtype.is_timestamp():
        return arg.dt.truncate("1us").cast(pl.Time)
    return arg


@translate.register(ops.Union)
def execute_union(op, **kw):
    result = pl.concat([translate(op.left, **kw), translate(op.right, **kw)])
    if op.distinct:
        return result.unique()
    return result


@translate.register(ops.Hash)
def execute_hash(op, **kw):
    return translate(op.arg, **kw).hash()


def _arg_min_max(op, func, **kw):
    key = op.key
    arg = op.arg

    if (op_where := op.where) is not None:
        key = ops.IfElse(op_where, key, None)
        arg = ops.IfElse(op_where, arg, None)

    translate_arg = translate(arg, **kw)
    translate_key = translate(key, **kw)

    not_null_mask = translate_arg.is_not_null() & translate_key.is_not_null()
    return translate_arg.filter(not_null_mask).gather(
        func(translate_key.filter(not_null_mask))
    )


@translate.register(ops.ArgMax)
def execute_arg_max(op, **kw):
    return _arg_min_max(op, pl.Expr.arg_max, **kw)


@translate.register(ops.ArgMin)
def execute_arg_min(op, **kw):
    return _arg_min_max(op, pl.Expr.arg_min, **kw)


@translate.register(ops.SQLStringView)
def execute_sql_string_view(op, *, ctx: pl.SQLContext, **kw):
    child = translate(op.child, ctx=ctx, **kw)
    ctx.register(op.name, child)
    return ctx.execute(op.query)


@translate.register(ops.View)
def execute_view(op, *, ctx: pl.SQLContext, **kw):
    child = translate(op.child, ctx=ctx, **kw)
    ctx.register(op.name, child)
    return child


@translate.register(ops.SelfReference)
def execute_self_reference(op, **kw):
    return translate(op.parent, **kw)


@translate.register(ops.JoinTable)
def execute_join_table(op, **kw):
    return translate(op.parent, **kw)


@translate.register(ops.CountDistinctStar)
def execute_count_distinct_star(op, **kw):
    arg = pl.struct(*op.arg.schema.names)
    if op.where is not None:
        arg = arg.filter(translate(op.where, **kw))
    return arg.n_unique()


_UDF_INVOKERS = {
    # Convert polars series into a list
    #   -> map the function element by element
    #   -> convert back to a polars series
    InputType.PYTHON: lambda func, dtype, args: pl.Series(
        map(func, *(arg.to_list() for arg in args)),
        dtype=PolarsType.from_ibis(dtype),
    ),
    # Convert polars series into a pyarrow array
    #  -> invoke the function on the pyarrow array
    #  -> cast the result to match the ibis dtype
    #  -> convert back to a polars series
    InputType.PYARROW: lambda func, dtype, args: pl.from_arrow(
        func(*(arg.to_arrow() for arg in args)).cast(dtype.to_pyarrow()),
    ),
}


@translate.register(ops.ScalarUDF)
def execute_scalar_udf(op, **kw):
    input_type = op.__input_type__
    if input_type in _UDF_INVOKERS:
        dtype = op.dtype
        return pl.map_batches(
            exprs=[translate(arg, **kw) for arg in op.args],
            function=partial(_UDF_INVOKERS[input_type], op.__func__, dtype),
            return_dtype=PolarsType.from_ibis(dtype),
        )
    elif input_type == InputType.BUILTIN:
        first, *rest = map(translate, op.args)
        return getattr(first, op.__func_name__)(*rest)
    else:
        raise NotImplementedError(
            f"UDF input type {input_type} not supported for Polars"
        )


@translate.register(ops.AggUDF)
def execute_agg_udf(op, **kw):
    args = (arg for name, arg in zip(op.argnames, op.args) if name != "where")
    first, *rest = map(partial(translate, **kw), args)
    if (where := op.where) is not None:
        first = first.filter(translate(where, **kw))
    return getattr(first, op.__func_name__)(*rest)


@translate.register(ops.RegexSplit)
def execute_regex_split(op, **kw):
    import pyarrow.compute as pc

    def split(args):
        arg, patterns = args
        if len(patterns) != 1:
            raise com.IbisError(
                "Only a single scalar pattern is supported for Polars re_split"
            )
        return pl.from_arrow(pc.split_pattern_regex(arg.to_arrow(), patterns[0]))

    arg = translate(op.arg, **kw)
    pattern = translate(op.pattern, **kw)
    return pl.map_batches(
        exprs=(arg, pattern),
        function=split,
        return_dtype=PolarsType.from_ibis(op.dtype),
    )


@translate.register(ops.IntegerRange)
def execute_integer_range(op, **kw):
    if not isinstance(op.step, ops.Literal):
        raise com.UnsupportedOperationError(
            "Dynamic integer step not supported by Polars"
        )
    step = op.step.value

    dtype = PolarsType.from_ibis(op.dtype)
    empty = pl.int_ranges(0, 0, dtype=dtype)

    if step == 0:
        return empty

    start = translate(op.start, **kw)
    stop = translate(op.stop, **kw)
    return pl.int_ranges(start, stop, step, dtype=dtype)


@translate.register(ops.TimestampRange)
def execute_timestamp_range(op, **kw):
    if not isinstance(op.step, ops.Literal):
        raise com.UnsupportedOperationError(
            "Dynamic interval step not supported by Polars"
        )
    step = op.step.value
    unit = op.step.dtype.unit.value

    start = translate(op.start, **kw)
    stop = translate(op.stop, **kw)
    return pl.datetime_ranges(start, stop, f"{step}{unit}", closed="left")
