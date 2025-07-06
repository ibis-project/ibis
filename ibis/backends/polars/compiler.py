from __future__ import annotations

import calendar
import datetime
import math
import operator
from collections.abc import Mapping
from functools import partial, reduce, singledispatch
from math import isnan

import polars as pl
import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.polars.rewrites import PandasAsofJoin, PandasJoin, PandasRename
from ibis.backends.sql.compilers.base import STAR
from ibis.backends.sql.dialects import Polars
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
def translate(expr, **_):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op, **_):
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate.register(ops.Alias)
def alias(op, **kw):
    return translate(op.arg, **kw).alias(op.name)


@translate.register(ops.DatabaseTable)
def table(op, **_):
    return op.source._tables[op.name]


@translate.register(ops.DummyTable)
def dummy_table(op, **kw):
    selections = [translate(arg, **kw).alias(name) for name, arg in op.values.items()]
    return pl.DataFrame().lazy().select(selections)


@translate.register(ops.InMemoryTable)
def in_memory_table(op, *, ctx, **_):
    sql = sg.select(STAR).from_(sg.to_identifier(op.name, quoted=True)).sql(Polars)
    return ctx.execute(sql, eager=False)


def _make_duration(value, dtype):
    kwargs = {f"{dtype.resolution}s": value}
    return pl.duration(**kwargs)


@translate.register(ops.Literal)
def literal(op, **_):
    value = op.value
    dtype = op.dtype

    if value is None:
        return pl.lit(None, dtype=PolarsType.from_ibis(dtype))

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
    elif dtype.is_null() or dtype.is_binary():
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

        if dtype.is_numeric():
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


@translate.register(ops.Project)
def project(op, **kw):
    lf = translate(op.parent, **kw)

    selections = []
    unnests = []
    scalars = []
    for name, arg in op.values.items():
        if isinstance(arg, ops.Unnest):
            translated = translate(arg.arg, **kw).alias(name)
            unnests.append(name)
            selections.append(translated)
        elif isinstance(arg, ops.Value):
            translated = translate(arg, **kw).alias(name)
            if arg.shape.is_scalar():
                scalars.append(translated)
                selections.append(pl.col(name))
            else:
                selections.append(translated)
        else:
            raise com.TranslationError(
                "Polars backend is unable to compile selection with "
                f"operation type of {type(arg)}"
            )

    if scalars:
        # Scalars need to first be projected to columns with `with_columns`,
        # otherwise if a downstream select only selects out the scalar-backed
        # columns they'll return with only a single row.
        lf = lf.with_columns(scalars)

    if selections:
        lf = lf.select(selections)

        if unnests:
            lf = lf.explode(*unnests)

    return lf


@translate.register(ops.Sort)
def sort(op, **kw):
    lf = translate(op.parent, **kw)
    if not op.keys:
        return lf

    newcols = {gen_name("sort_key"): translate(col.expr, **kw) for col in op.keys}
    lf = lf.with_columns(**newcols)

    by = list(newcols.keys())
    descending = [key.descending for key in op.keys]
    nulls_last = [not key.nulls_first for key in op.keys]

    lf = lf.sort(by, descending=descending, nulls_last=nulls_last)

    return lf.drop(*by)


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
        func = (
            lf.with_columns(
                [translate(arg, **kw).alias(name) for name, arg in op.groups.items()]
            )
            .group_by(list(op.groups.keys()))
            .agg
        )
    else:
        func = lf.select

    if op.metrics:
        metrics = [
            translate(arg, in_group_by=bool(op.groups), **kw).alias(name)
            for name, arg in op.metrics.items()
        ]
        return func(metrics)

    return func()


@translate.register(PandasRename)
def rename(op, **kw):
    parent = translate(op.parent, **kw)
    return parent.rename(op.mapping)


@translate.register(PandasJoin)
def join(op, **kw):
    how = op.how
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)

    if how == "positional":
        return pl.concat([left, right], how="horizontal")

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
    elif how == "outer":
        how = "full"

    joined = left.join(right, on=on, how=how, coalesce=False)
    joined = joined.drop(*on)
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
    joined = joined.drop(*on, *by)
    return joined


@translate.register(ops.DropNull)
def drop_null(op, **kw):
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


@translate.register(ops.FillNull)
def fill_null(op, **kw):
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
    if op.value.dtype.is_struct():
        raise com.UnsupportedOperationError(
            "polars doesn't support contains with struct elements"
        )

    needle = translate(op.needle, **kw)
    value = translate(op.value, **kw)
    (rel,) = op.value.relations
    # The `collect` triggers computation, but there appears to be no way to
    # spell this operation in a polars-native way that's
    #
    # 1. not deprecated
    # 2. operates only using pl.Expr objects and methods
    #
    # In other words, we need to either rearchitect the polars compiler to
    # operate only with DataFrames, or compute first. I chose computing first
    # since it is less effort and we don't know how impactful it is.
    value = translate(rel, **kw).select(value).collect().to_series()

    return needle.map_batches(
        lambda needle, value=value: needle.is_in(value),
        return_dtype=pl.Boolean(),
    )


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
    return arg.str.len_bytes()


@translate.register(ops.Capitalize)
def capitalize(op, **kw):
    arg = translate(op.arg, **kw)
    first = arg.str.slice(0, 1).str.to_uppercase()
    rest = arg.str.slice(1, None).str.to_lowercase()
    return first + rest


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
    return arg.str.reverse()


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


@translate.register(ops.ArrayStringJoin)
def array_string_join(op, **kw):
    arg = translate(op.arg, **kw)
    sep = _literal_value(op.sep)
    return pl.when(arg.list.len() > 0).then(arg.list.join(sep)).otherwise(None)


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

    return arg.clip(lower, upper)


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
    ops.ApproxCountDistinct: "approx_n_unique",
    ops.Count: "count",
    ops.CountDistinct: "n_unique",
    ops.Max: "max",
    ops.Mean: "mean",
    ops.Median: "median",
    ops.Min: "min",
    ops.Sum: "sum",
    ops.BitOr: "bitwise_or",
    ops.BitAnd: "bitwise_and",
    ops.BitXor: "bitwise_xor",
}


def execute_reduction(op, **kw):
    arg = translate(op.arg, **kw)

    if op.where is not None:
        arg = arg.filter(translate(op.where, **kw))

    method = _reductions[type(op)]

    return getattr(arg, method)()


for cls in _reductions:
    translate.register(cls, execute_reduction)


@translate.register(ops.Sum)
def execute_sum(op, **kw):
    arg = translate(op.arg, **kw)
    if (where := op.where) is not None:
        arg = arg.filter(translate(where, **kw))
    return pl.when(arg.count() > 0).then(arg.sum()).otherwise(None)


@translate.register(ops.First)
@translate.register(ops.Last)
@translate.register(ops.Arbitrary)
def execute_first_last(op, **kw):
    arg = translate(op.arg, **kw)

    predicate = True if getattr(op, "include_null", False) else arg.is_not_null()
    if op.where is not None:
        predicate &= translate(op.where, **kw)

    arg = arg.filter(predicate)

    if order_by := getattr(op, "order_by", ()):
        keys = [translate(k.expr, **kw).filter(predicate) for k in order_by]
        descending = [k.descending for k in order_by]
        arg = arg.sort_by(keys, descending=descending)

    return arg.last() if isinstance(op, ops.Last) else arg.first()


@translate.register(ops.StandardDev)
@translate.register(ops.Variance)
def execute_std_var(op, **kw):
    arg = translate(op.arg, **kw)

    if op.where is not None:
        arg = arg.filter(translate(op.where, **kw))

    method = "std" if isinstance(op, ops.StandardDev) else "var"
    ddof = 0 if op.how == "pop" else 1

    return getattr(arg, method)(ddof=ddof)


@translate.register(ops.Kurtosis)
def execute_kurtosis(op, **kw):
    arg = translate(op.arg, **kw)

    if (where := op.where) is not None:
        arg = arg.filter(translate(where, **kw))

    return arg.kurtosis(bias=op.how == "pop")


@translate.register(ops.Mode)
def execute_mode(op, **kw):
    arg = translate(op.arg, **kw)

    predicate = arg.is_not_null()
    if (where := op.where) is not None:
        predicate &= translate(where, **kw)

    # `mode` can return more than one value so the additional `get(0)` call is
    # necessary to enforce aggregation behavior of a scalar value per group
    #
    # eventually we may want to support an Ibis API like `modes` that returns a
    # list of all the modes per group.
    return arg.filter(predicate).mode().get(0)


@translate.register(ops.Quantile)
@translate.register(ops.ApproxQuantile)
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


@translate.register(ops.Sample)
def sample(op, **kw):
    if op.seed is not None:
        raise com.UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )
    table = translate(op.parent, **kw)
    # Disable predicate pushdown since `t.filter(...).sample(...)` could have
    # different statistical or performance characteristics than
    # `t.sample(...).filter(...)`. Same for slice pushdown.
    return table.map_batches(
        lambda df: df.sample(fraction=op.fraction),
        predicate_pushdown=False,
        slice_pushdown=False,
        streamable=True,
    )


@translate.register(ops.CountStar)
def count_star(op, **kw):
    if (where := op.where) is not None:
        condition = translate(where, **kw)
        result = condition.sum()
    else:
        result = pl.len()
    return result.cast(PolarsType.from_ibis(op.dtype))


@translate.register(ops.TimestampNow)
def timestamp_now(op, **_):
    return pl.lit(datetime.datetime.now())


@translate.register(ops.DateNow)
def date_now(op, **_):
    return pl.lit(datetime.date.today())


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
    return arg.dt.truncate(unit)


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
    res = arg.dt.truncate(interval)
    if offset is not None:
        res = res.dt.offset_by(offset)
    return res


@translate.register(ops.DateFromYMD)
def date_from_ymd(op, **kw):
    return pl.date(
        year=translate(op.year, **kw),
        month=translate(op.month, **kw),
        day=translate(op.day, **kw),
    )


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


@translate.register(ops.StringToDate)
def string_to_date(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.strptime(
        dtype=pl.Date,
        format=_literal_value(op.format_str),
    )


@translate.register(ops.StringToTime)
def string_to_time(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.str.to_time(format=_literal_value(op.format_str))


@translate.register(ops.StringToTimestamp)
def string_to_timestamp(op, **kw):
    arg = translate(op.arg, **kw)
    format = _literal_value(op.format_str)
    return arg.str.strptime(dtype=pl.Datetime, format=format)


@translate.register(ops.TimestampDiff)
def timestamp_diff(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    # TODO: truncating both to seconds is necessary to conform to the output
    # type of the operation
    return left.dt.truncate("1s") - right.dt.truncate("1s")


@translate.register(ops.ArraySort)
def array_sort(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.list.sort()


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
    # Workaround for https://github.com/pola-rs/polars/issues/17294
    # pl.concat_list(Iterable[T]) results in pl.List[T], EXCEPT when T is a
    # pl.List, in which case pl.concat_list(Iterable[pl.List[T]]) results in pl.List[T].
    # If polars ever supports a more consistent array constructor,
    # we should switch to that.
    if op.dtype.value_type.is_array():
        cols = [c.implode() for c in cols]
    return pl.concat_list(cols)


@translate.register(ops.ArrayCollect)
def array_collect(op, in_group_by=False, **kw):
    arg = translate(op.arg, **kw)

    predicate = True if op.include_null else arg.is_not_null()
    if op.where is not None:
        predicate &= translate(op.where, **kw)

    arg = arg.filter(predicate)

    if op.order_by:
        keys = [translate(k.expr, **kw).filter(predicate) for k in op.order_by]
        descending = [k.descending for k in op.order_by]
        arg = arg.sort_by(keys, descending=descending, nulls_last=True)

    if op.distinct:
        arg = arg.unique(maintain_order=op.order_by is not None)

    # Polars' behavior changes for `implode` within a `group_by` currently.
    # See https://github.com/pola-rs/polars/issues/16756
    return arg if in_group_by else arg.implode()


@translate.register(ops.ArrayFlatten)
def array_flatten(op, **kw):
    result = translate(op.arg, **kw)
    return (
        pl.when(result.is_null())
        .then(None)
        .when(result.list.len() == 0)
        .then([])
        # polars doesn't have an efficient API (yet?) for removing one level of
        # nesting from an array so we use elementwise evaluation
        #
        # https://github.com/ibis-project/ibis/issues/10135
        .otherwise(result.list.eval(pl.element().flatten()))
    )


_date_methods = {
    ops.ExtractDay: "day",
    ops.ExtractMonth: "month",
    ops.ExtractYear: "year",
    ops.ExtractIsoYear: "iso_year",
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
    return method(arg.dt)


@translate.register(ops.ExtractEpochSeconds)
def extract_epoch_seconds(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.dt.epoch("s")


_unary = {
    # TODO(kszucs): factor out the lambdas
    ops.Abs: operator.methodcaller("abs"),
    ops.Acos: operator.methodcaller("arccos"),
    ops.Asin: operator.methodcaller("arcsin"),
    ops.Atan: operator.methodcaller("arctan"),
    ops.Ceil: lambda arg: arg.ceil().cast(pl.Int64),
    ops.Cos: operator.methodcaller("cos"),
    ops.Cot: lambda arg: 1.0 / arg.tan(),
    ops.DayOfWeekIndex: lambda arg: arg.dt.weekday() - 1,
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
    ops.BitwiseNot: operator.invert,
}


@translate.register(ops.DayOfWeekName)
def day_of_week_name(op, **kw):
    index = translate(op.arg, **kw).dt.weekday() - 1
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
    arg_dtype = op_arg.dtype

    arg = translate(op_arg, **kw)

    dtype = PolarsType.from_ibis(arg_dtype)

    lower_bound = op.lower_bound
    lower = translate(lower_bound, **kw)

    if lower_bound.dtype != arg_dtype:
        lower = lower.cast(dtype)

    upper_bound = op.upper_bound
    upper = translate(upper_bound, **kw)

    if upper_bound.dtype != arg_dtype:
        upper = upper.cast(dtype)

    return arg.is_between(lower, upper, closed="both")


@translate.register(ops.BitwiseLeftShift)
def bitwise_left_shift(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.cast(pl.Int64) * 2 ** right.cast(pl.Int64)


@translate.register(ops.BitwiseRightShift)
def bitwise_right_shift(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.cast(pl.Int64) // 2 ** right.cast(pl.Int64)


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
    ops.BitwiseOr: operator.or_,
    ops.BitwiseXor: operator.xor,
    ops.BitwiseAnd: operator.and_,
    ops.Modulus: operator.mod,
    ops.Atan2: pl.arctan2,
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
    return pl.lit(math.e)


@translate.register(ops.Pi)
def execute_pi(op, **_):
    return pl.lit(math.pi)


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


@translate.register(ops.Intersection)
def execute_intersection(op, *, ctx, **kw):
    left = gen_name("polars_intersect_left")
    right = gen_name("polars_intersect_right")

    ctx.register_many(
        frames={
            left: translate(op.left, ctx=ctx, **kw),
            right: translate(op.right, ctx=ctx, **kw),
        }
    )

    sql = (
        sg.select(STAR)
        .from_(sg.to_identifier(left, quoted=True))
        .intersect(sg.select(STAR).from_(sg.to_identifier(right, quoted=True)))
    )

    result = ctx.execute(sql.sql(Polars), eager=False)

    if op.distinct is True:
        return result.unique()
    return result


@translate.register(ops.Difference)
def execute_difference(op, *, ctx, **kw):
    left = gen_name("polars_diff_left")
    right = gen_name("polars_diff_right")

    ctx.register_many(
        frames={
            left: translate(op.left, ctx=ctx, **kw),
            right: translate(op.right, ctx=ctx, **kw),
        }
    )

    sql = (
        sg.select(STAR)
        .from_(sg.to_identifier(left, quoted=True))
        .except_(sg.select(STAR).from_(sg.to_identifier(right, quoted=True)))
    )

    result = ctx.execute(sql.sql(Polars), eager=False)

    if op.distinct is True:
        return result.unique()
    return result


@translate.register(ops.Hash)
def execute_hash(op, **kw):
    # polars' hash() returns a uint64, but we want to return an int64
    return translate(op.arg, **kw).hash().reinterpret(signed=True)


def _arg_min_max(op, func, **kw):
    key = translate(op.key, **kw)
    arg = translate(op.arg, **kw)

    if op.where is not None:
        where = translate(op.where, **kw)
        arg = arg.filter(where)
        key = key.filter(where)

    return arg.get(func(key))


@translate.register(ops.ArgMax)
def execute_arg_max(op, **kw):
    return _arg_min_max(op, pl.Expr.arg_max, **kw)


@translate.register(ops.ArgMin)
def execute_arg_min(op, **kw):
    return _arg_min_max(op, pl.Expr.arg_min, **kw)


@translate.register(ops.SQLStringView)
def execute_sql_string_view(op, *, ctx: pl.SQLContext, **kw):
    translate(op.child, ctx=ctx, **kw)
    return ctx.execute(op.query)


@translate.register(ops.View)
def execute_view(op, *, ctx: pl.SQLContext, **kw):
    child = translate(op.child, ctx=ctx, **kw)
    ctx.register(op.name, child)
    return child


@translate.register(ops.Reference)
def execute_reference(op, **kw):
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
    InputType.PYTHON: lambda func, dtype, fields, args: pl.Series(
        map(func, *(arg.to_list() for arg in map(args.struct.field, fields))),
        dtype=PolarsType.from_ibis(dtype),
    ),
    # Convert polars series into a pyarrow array
    #  -> invoke the function on the pyarrow array
    #  -> cast the result to match the ibis dtype
    #  -> convert back to a polars series
    InputType.PYARROW: lambda func, dtype, fields, args: pl.from_arrow(
        func(*(arg.to_arrow() for arg in map(args.struct.field, fields))).cast(
            dtype.to_pyarrow()
        ),
    ),
}


@translate.register(ops.ScalarUDF)
def execute_scalar_udf(op, **kw):
    input_type = op.__input_type__
    if input_type in _UDF_INVOKERS:
        dtype = op.dtype
        argnames = op.argnames
        args = pl.struct(
            **dict(zip(argnames, (translate(arg, **kw) for arg in op.args)))
        )
        return args.map_batches(
            function=partial(_UDF_INVOKERS[input_type], op.__func__, dtype, argnames),
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

    dtype = PolarsType.from_ibis(op.dtype.value_type)
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


@translate.register(ops.TimeFromHMS)
def execute_time_from_hms(op, **kw):
    return pl.time(
        hour=translate(op.hours, **kw),
        minute=translate(op.minutes, **kw),
        second=translate(op.seconds, **kw),
    )


@translate.register(ops.DropColumns)
def execute_drop_columns(op, **kw):
    parent = translate(op.parent, **kw)
    return parent.drop(op.columns_to_drop)


@translate.register(ops.ArraySum)
def execute_array_agg(op, **kw):
    arg = translate(op.arg, **kw)
    # workaround polars annoying sum([]) == 0 behavior
    #
    # the polars behavior is consistent with math, but inconsistent
    # with every other query engine every built.
    no_nulls = arg.list.drop_nulls()
    return pl.when(no_nulls.list.len() == 0).then(None).otherwise(no_nulls.list.sum())


@translate.register(ops.ArrayMean)
def execute_array_mean(op, **kw):
    return translate(op.arg, **kw).list.mean()


@translate.register(ops.ArrayMin)
def execute_array_min(op, **kw):
    return translate(op.arg, **kw).list.min()


@translate.register(ops.ArrayMax)
def execute_array_max(op, **kw):
    return translate(op.arg, **kw).list.max()


@translate.register(ops.ArrayAny)
def execute_array_any(op, **kw):
    arg = translate(op.arg, **kw)
    no_nulls = arg.list.drop_nulls()
    return pl.when(no_nulls.list.len() == 0).then(None).otherwise(no_nulls.list.any())


@translate.register(ops.ArrayAll)
def execute_array_all(op, **kw):
    arg = translate(op.arg, **kw)
    no_nulls = arg.list.drop_nulls()
    return pl.when(no_nulls.list.len() == 0).then(None).otherwise(no_nulls.list.all())


@translate.register(ops.GroupConcat)
def execute_group_concat(op, **kw):
    arg = translate(op.arg, **kw)
    sep = _literal_value(op.sep)

    predicate = arg.is_not_null()

    if (where := op.where) is not None:
        predicate &= translate(where, **kw)

    arg = arg.filter(predicate)

    if order_by := op.order_by:
        keys = [translate(k.expr, **kw).filter(predicate) for k in order_by]
        descending = [k.descending for k in order_by]
        arg = arg.sort_by(keys, descending=descending)

    return pl.when(arg.count() > 0).then(arg.str.join(sep)).otherwise(None)


@translate.register(ops.DateDelta)
def execute_date_delta(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    delta = left - right
    method_name = f"total_{_literal_value(op.part)}s"
    return getattr(delta.dt, method_name)()


@translate.register(ops.ArrayIndex)
def execute_array_index(op, **kw):
    arg = translate(op.arg, **kw)
    index = translate(op.index, **kw)
    return arg.list.get(index)


@translate.register(ops.ArraySlice)
def visit_ArraySlice(op, **kw):
    arg = translate(op.arg, **kw)
    arg_length = arg.list.len()

    start = translate(op.start, **kw) if op.start is not None else 0
    stop = translate(op.stop, **kw) if op.stop is not None else arg_length

    def normalize_index(index, length):
        index = pl.when(index < 0).then(length + index).otherwise(index)
        return (
            pl.when(index < 0)
            .then(0)
            .when(index > length)
            .then(length)
            .otherwise(index)
        )

    start = normalize_index(start, arg_length)
    stop = normalize_index(stop, arg_length)

    slice_len = pl.when(stop > start).then(stop - start).otherwise(0)

    return arg.list.slice(start, slice_len)


@translate.register(ops.ArrayContains)
def visit_ArrayContains(op, **kw):
    arg = translate(op.arg, **kw)
    value = translate(op.other, **kw)
    return arg.list.contains(value)


@translate.register(ops.ArrayRemove)
def visit_ArrayRemove(op, **kw):
    arg = translate(op.arg, **kw)
    value = _literal_value(op.other)
    return arg.list.set_difference(pl.lit([value]))


@translate.register(ops.ArrayUnion)
def visit_ArrayUnion(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.list.set_union(right)


@translate.register(ops.ArrayDistinct)
def visit_ArrayDistinct(op, **kw):
    arg = translate(op.arg, **kw)
    return arg.list.unique()


@translate.register(ops.ArrayIntersect)
def visit_ArrayIntersect(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    return left.list.set_intersection(right)


@translate.register(ops.StringFind)
def visit_StringFind(op, **kw):
    arg = translate(op.arg, **kw)
    start = translate(op.start, **kw) if op.start is not None else 0
    end = translate(op.end, **kw) if op.end is not None else None
    expr = arg.str.slice(start, end).str.find(_literal_value(op.substr), literal=True)
    return pl.when(expr.is_null()).then(-1).otherwise(expr + start)
