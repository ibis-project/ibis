from __future__ import annotations

import collections
import enum
import functools
import operator
from functools import partial, reduce

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as pt
import toolz
from packaging.version import parse as vparse
from pyspark.sql import Window
from pyspark.sql.functions import PandasUDFType, pandas_udf

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import interval
from ibis.backends.base.df.timecontext import adjust_context
from ibis.backends.pandas.execution import execute
from ibis.backends.pyspark.datatypes import PySparkType
from ibis.backends.pyspark.timecontext import (
    combine_time_context,
    filter_by_time_context,
)
from ibis.common.collections import frozendict
from ibis.config import options
from ibis.expr.operations.udf import InputType
from ibis.util import any_of, guid


class AggregationContext(enum.Enum):
    ENTIRE = 0
    WINDOW = 1
    GROUP = 2


class PySparkExprTranslator:
    _registry = {}

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def translate(self, op, *, scope, timecontext, **kwargs):
        """Translate Ibis expression into a PySpark object.

        All translated expressions are cached within scope. If an expression is
        found within scope, it's returned. Otherwise, the it's translated and
        cached for future reference.

        Parameters
        ----------
        op
            An ibis operation.
        scope
            dictionary mapping from operation to translated result
        timecontext
            time context associated with expr
        kwargs
            parameters passed as keyword args

        Returns
        -------
        pyspark.sql.DataFrame
            translated PySpark DataFrame or Column object
        """
        # TODO(cpcloud): remove the udf instance checking when going to sqlglot
        if isinstance(op, ops.ScalarUDF):
            formatter = compile_scalar_udf
            result = formatter(self, op, scope=scope, timecontext=timecontext, **kwargs)
            return result
        elif (
            not isinstance(op, ops.ScalarParameter)
            and (result := scope.get_value(op, timecontext)) is not None
        ):
            return result
        elif (formatter := self._registry.get(type(op))) is not None:
            result = formatter(self, op, scope=scope, timecontext=timecontext, **kwargs)
            scope.set_value(op, timecontext, result)
            return result
        else:
            raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


compiles = PySparkExprTranslator.compiles


# TODO(kszucs): there are plenty of repetitions in this file which should be
# reduced at some point


@compiles(ops.DatabaseTable)
def compile_datasource(t, op, *, timecontext, **kwargs):
    df = op.source._session.table(op.name)
    return filter_by_time_context(df, timecontext).alias(op.name)


@compiles(ops.SQLQueryResult)
def compile_sql_query_result(t, op, **kwargs):
    query, _, client = op.args
    return client._session.sql(query)


def _can_be_replaced_by_column_name(column, table):
    """Return whether the given `column` can be replaced by its literal name.

    `True` when `column` and `table[column.get_name()]` are semantically equivalent.
    """
    # Each check below is necessary to distinguish a pure projection from
    # other valid selections, such as a mutation that assigns a new column
    # or changes the value of an existing column.
    return (
        isinstance(column, ops.TableColumn)
        and column.table == table
        and column.name in table.schema
        # TODO(kszucs): do we really need this condition?
        and column == table.to_expr()[column.name].op()
    )


@compiles(ops.Alias)
def compile_alias(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return arg.alias(op.name)


@compiles(ops.Selection)
def compile_selection(t, op, *, scope, timecontext, **kwargs):
    # In selection, there could be multiple children that point to the
    # same root table. e.g. window with different sizes on a table.
    # We need to get the 'combined' time range that is a superset of every
    # time context among child nodes, and pass this as context to
    # source table to get all data within time context loaded.
    arg_timecontexts = [
        adjust_context(node, scope=scope, timecontext=timecontext)  # , **kwargs)
        for node in op.selections
        if timecontext
    ]
    adjusted_timecontext = combine_time_context(arg_timecontexts)
    # If this is a sort or filter node, op.selections is empty
    # in this case, we use the original timecontext
    if not adjusted_timecontext:
        adjusted_timecontext = timecontext
    src_table = t.translate(
        op.table, scope=scope, timecontext=adjusted_timecontext, **kwargs
    )

    col_in_selection_order = []
    col_to_drop = []
    result_table = src_table

    for predicate in op.predicates:
        col = t.translate(predicate, scope=scope, timecontext=timecontext, **kwargs)
        # Due to an upstream Spark issue (SPARK-33057) we cannot
        # directly use filter with a window operation. The workaround
        # here is to assign a temporary column for the filter predicate,
        # do the filtering, and then drop the temporary column.
        filter_column = f"predicate_{guid()}"
        result_table = result_table.withColumn(filter_column, col)
        result_table = result_table.filter(F.col(filter_column))
        result_table = result_table.drop(filter_column)

    for selection in op.selections:
        if isinstance(selection, ops.TableNode):
            col_in_selection_order.extend(selection.schema.names)
        elif isinstance(selection, ops.Value):
            # If the selection is a straightforward projection of a table
            # column from the root table itself (i.e. excluding mutations and
            # renames), we can get the selection name directly.
            if _can_be_replaced_by_column_name(selection, op.table):
                col_in_selection_order.append(selection.name)
            else:
                col = t.translate(
                    selection,
                    scope=scope,
                    timecontext=adjusted_timecontext,
                    **kwargs,
                )
                col = col.alias(selection.name)
                col_in_selection_order.append(col)
        else:
            raise NotImplementedError(
                f"Unrecognized type in selections: {type(selection)}"
            )
    if col_in_selection_order:
        result_table = result_table[col_in_selection_order]

    if col_to_drop:
        result_table = result_table.drop(*col_to_drop)

    if op.sort_keys:
        sort_cols = [
            t.translate(key, scope=scope, timecontext=timecontext, **kwargs)
            for key in op.sort_keys
        ]
        result_table = result_table.sort(*sort_cols)

    return filter_by_time_context(result_table, timecontext, adjusted_timecontext)


@compiles(ops.SortKey)
def compile_sort_desc(t, op, **kwargs):
    col = t.translate(op.expr, **kwargs)
    return col.asc() if op.ascending else col.desc()


def compile_nan_as_null(compile_func):
    @functools.wraps(compile_func)
    def wrapper(t, op, *args, **kwargs):
        compiled = compile_func(t, op, *args, **kwargs)
        if options.pyspark.treat_nan_as_null and isinstance(op.dtype, dt.Floating):
            return F.nanvl(compiled, F.lit(None))
        else:
            return compiled

    return wrapper


@compiles(ops.TableColumn)
@compile_nan_as_null
def compile_column(t, op, **kwargs):
    name = op.name
    table = op.table
    try:
        name = f"`{table.name}`.`{name}`"
    except AttributeError:
        spark_df = t.translate(table, **kwargs)
        return spark_df[name]
    else:
        return F.col(name)


@compiles(ops.StructField)
def compile_struct_field(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return arg[op.field]


@compiles(ops.StructColumn)
def compile_struct_column(t, op, **kwargs):
    return F.struct(
        *(
            t.translate(col, **kwargs).alias(name)
            for name, col in zip(op.names, op.values)
        )
    )


@compiles(ops.SelfReference)
def compile_self_reference(t, op, **kwargs):
    return t.translate(op.table, **kwargs).alias(op.name)


@compiles(ops.Cast)
def compile_cast(t, op, **kwargs):
    if op.to.is_interval():
        if isinstance(op.arg, ops.Literal):
            return interval(op.arg.value, op.to.unit).op()
        else:
            raise com.UnsupportedArgumentError(
                "Casting to intervals is only supported for literals "
                f"in the PySpark backend. {type(op.arg)} not allowed."
            )

    cast_type = PySparkType.from_ibis(op.to)

    src_column = t.translate(op.arg, **kwargs)
    return src_column.cast(cast_type)


@compiles(ops.Limit)
def compile_limit(t, op, **kwargs):
    if (n := op.n) is not None and not isinstance(n, int):
        raise com.UnsupportedArgumentError(
            "Dynamic LIMIT is not implemented upstream in PySpark"
        )
    if not isinstance(offset := op.offset, int):
        raise com.UnsupportedArgumentError(
            "Dynamic OFFSET is not implemented upstream in PySpark"
        )
    if n != 0 and offset != 0:
        raise com.UnsupportedArgumentError(
            "PySpark backend does not support non-zero offset values for "
            f"the limit operation. Got offset {offset:d}."
        )
    df = t.translate(op.table, **kwargs)

    if n is not None:
        return df.limit(n)
    else:
        return df


@compiles(ops.Sample)
def compile_sample(t, op, **kwargs):
    df = t.translate(op.table, **kwargs)
    return df.sample(fraction=op.fraction, seed=op.seed)


@compiles(ops.And)
def compile_and(t, op, **kwargs):
    return t.translate(op.left, **kwargs) & t.translate(op.right, **kwargs)


@compiles(ops.Or)
def compile_or(t, op, **kwargs):
    return t.translate(op.left, **kwargs) | t.translate(op.right, **kwargs)


@compiles(ops.Xor)
def compile_xor(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return (left | right) & ~(left & right)


@compiles(ops.Equals)
def compile_equals(t, op, **kwargs):
    return t.translate(op.left, **kwargs) == t.translate(op.right, **kwargs)


@compiles(ops.Not)
def compile_not(t, op, **kwargs):
    return ~t.translate(op.arg, **kwargs)


@compiles(ops.NotEquals)
def compile_not_equals(t, op, **kwargs):
    return t.translate(op.left, **kwargs) != t.translate(op.right, **kwargs)


@compiles(ops.Greater)
def compile_greater(t, op, **kwargs):
    return t.translate(op.left, **kwargs) > t.translate(op.right, **kwargs)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, op, **kwargs):
    return t.translate(op.left, **kwargs) >= t.translate(op.right, **kwargs)


@compiles(ops.Less)
def compile_less(t, op, **kwargs):
    return t.translate(op.left, **kwargs) < t.translate(op.right, **kwargs)


@compiles(ops.LessEqual)
def compile_less_equal(t, op, **kwargs):
    return t.translate(op.left, **kwargs) <= t.translate(op.right, **kwargs)


@compiles(ops.Between)
def compile_between(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    lower_bound = t.translate(op.lower_bound, **kwargs)
    upper_bound = t.translate(op.upper_bound, **kwargs)
    return arg.between(lower_bound, upper_bound)


@compiles(ops.Multiply)
def compile_multiply(t, op, **kwargs):
    return t.translate(op.left, **kwargs) * t.translate(op.right, **kwargs)


@compiles(ops.Subtract)
def compile_subtract(t, op, **kwargs):
    return t.translate(op.left, **kwargs) - t.translate(op.right, **kwargs)


@compiles(ops.Literal)
@compile_nan_as_null
def compile_literal(t, op, *, raw=False, **kwargs):
    """If raw is True, don't wrap the result with F.lit()."""

    value = op.value
    dtype = op.dtype

    if value is None:
        return F.lit(None)

    if raw:
        return value

    if dtype.is_interval():
        # execute returns a Timedelta and value is nanoseconds
        return execute(op).value

    if isinstance(value, collections.abc.Set):
        # Don't wrap set with F.lit
        if isinstance(value, frozenset):
            # Spark doesn't like frozenset
            return set(value)
        else:
            return value
    elif dtype.is_array():
        return F.array(*map(F.lit, value))
    elif dtype.is_struct():
        return F.struct(*(F.lit(val).alias(name) for name, val in value.items()))
    elif dtype.is_timestamp():
        return F.from_utc_timestamp(F.lit(str(value)), tz="UTC")
    elif dtype.is_decimal():
        return F.lit(value.normalize())
    else:
        return F.lit(value)


@compiles(ops.Aggregation)
def compile_aggregation(t, op, **kwargs):
    src_table = t.translate(op.table, **kwargs)

    if op.having:
        raise com.UnsupportedOperationError(
            "The PySpark backend does not support `having` because the underlying "
            "PySpark API does not support it. Use a filter on the aggregation "
            "expression instead."
        )

    if op.predicates:
        predicate = reduce(ops.And, op.predicates)
        src_table = src_table.filter(t.translate(predicate, **kwargs))

    if op.by:
        aggcontext = AggregationContext.GROUP
        bys = [t.translate(b, **kwargs) for b in op.by]
        src_table = src_table.groupby(*bys)
    else:
        aggcontext = AggregationContext.ENTIRE

    aggs = [t.translate(m, aggcontext=aggcontext, **kwargs) for m in op.metrics]
    return src_table.agg(*aggs)


@compiles(ops.Union)
def compile_union(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    result = left.union(right)
    return result.distinct() if op.distinct else result


@compiles(ops.Intersection)
def compile_intersection(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return left.intersect(right) if op.distinct else left.intersectAll(right)


@compiles(ops.Difference)
def compile_difference(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return left.subtract(right) if op.distinct else left.exceptAll(right)


@compiles(ops.InColumn)
def in_column(t, op, **kwargs):
    value = t.translate(op.value, **kwargs)
    options = t.translate(op.options, **kwargs)
    return value.isin(options)


@compiles(ops.InValues)
def in_values(t, op, **kwargs):
    value = t.translate(op.value, **kwargs)
    options = [t.translate(option, **kwargs) for option in op.options]
    return value.isin(options)


@compiles(ops.StartsWith)
def compile_startswith(t, op, **kwargs):
    col = t.translate(op.arg, **kwargs)
    start = t.translate(op.start, **kwargs)
    return col.startswith(start)


@compiles(ops.EndsWith)
def compile_endswith(t, op, **kwargs):
    col = t.translate(op.arg, **kwargs)
    end = t.translate(op.end, **kwargs)
    return col.endswith(end)


def _is_table(table):
    # TODO(kszucs): is has a pretty misleading name, should be removed
    try:
        return isinstance(table.arg, ops.TableNode)
    except AttributeError:
        return False


def compile_aggregator(
    t, op, *, fn, aggcontext=None, where_excludes: tuple[str, ...] = (), **kwargs
):
    if (where := getattr(op, "where", None)) is not None:
        condition = t.translate(where, **kwargs)
    else:
        condition = None

    def translate_arg(arg, include_where: bool):
        src_col = t.translate(arg, **kwargs)

        if include_where and condition is not None:
            src_col = F.when(condition, src_col)
        return src_col

    src_inputs = tuple(
        (argname, arg)
        for argname, arg in zip(op.argnames, op.args)
        if argname != "where"
    )
    src_cols = tuple(
        translate_arg(
            arg, include_where=(not where_excludes) or argname not in where_excludes
        )
        for argname, arg in src_inputs
        if isinstance(arg, ops.Node)
    )

    col = fn(*src_cols)
    if aggcontext:
        return col
    else:
        # We are trying to compile a expr such as some_col.max()
        # to a Spark expression.
        # Here we get the root table df of that column and compile
        # the expr to:
        # df.select(max(some_col))
        if _is_table(op):
            (src_col,) = src_cols
            return src_col.select(col)
        table_op = an.find_first_base_table(op)
        return t.translate(table_op, **kwargs).select(col)


@compiles(ops.GroupConcat)
def compile_group_concat(t, op, **kwargs):
    sep = t.translate(op.sep, raw=True, **kwargs)

    def fn(col, _):
        collected = F.collect_list(col)
        return F.array_join(
            F.when(F.size(collected) == 0, F.lit(None)).otherwise(collected),
            sep,
        )

    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.Any)
def compile_any(t, op, *, aggcontext=None, **kwargs):
    return compile_aggregator(t, op, fn=F.max, aggcontext=aggcontext, **kwargs)


@compiles(ops.All)
def compile_all(t, op, *args, **kwargs):
    return compile_aggregator(t, op, *args, fn=F.min, **kwargs)


@compiles(ops.Count)
def compile_count(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.count, **kwargs)


@compiles(ops.CountDistinct)
def compile_count_distinct(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.count_distinct, **kwargs)


@compiles(ops.CountStar)
def compile_count_star(t, op, aggcontext=None, **kwargs):
    src_table = t.translate(op.arg, **kwargs)

    src_col = F.lit(1)

    if (where := op.where) is not None:
        src_col = F.when(t.translate(where, **kwargs), src_col)

    col = F.count(src_col)
    if aggcontext is not None:
        return col
    else:
        return src_table.select(col)


@compiles(ops.CountDistinctStar)
def compile_count_distinct_star(t, op, aggcontext=None, **kwargs):
    src_table = t.translate(op.arg, **kwargs)
    src_col = F.struct(*map(F.col, op.arg.schema.names))

    if (where := op.where) is not None:
        src_col = F.when(t.translate(where, **kwargs), src_col)

    src_col = F.countDistinct(src_col)
    if aggcontext is not None:
        return src_col
    else:
        return src_table.select(src_col)


@compiles(ops.Max)
def compile_max(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.max, **kwargs)


@compiles(ops.Min)
def compile_min(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.min, **kwargs)


@compiles(ops.Mean)
def compile_mean(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.mean, **kwargs)


@compiles(ops.Sum)
def compile_sum(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.sum, **kwargs)


@compiles(ops.ApproxCountDistinct)
def compile_approx_count_distinct(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.approx_count_distinct, **kwargs)


@compiles(ops.ApproxMedian)
def compile_approx_median(t, op, **kwargs):
    return compile_aggregator(
        t, op, fn=lambda arg: F.percentile_approx(arg, 0.5), **kwargs
    )


@compiles(ops.StandardDev)
def compile_std(t, op, **kwargs):
    how = op.how

    if how == "sample":
        fn = F.stddev_samp
    elif how == "pop":
        fn = F.stddev_pop
    else:
        raise com.TranslationError(f"Unexpected 'how' in translation: {how}")

    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.Variance)
def compile_variance(t, op, **kwargs):
    how = op.how

    if how == "sample":
        fn = F.var_samp
    elif how == "pop":
        fn = F.var_pop
    else:
        raise com.TranslationError(f"Unexpected 'how' in translation: {how}")

    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.Covariance)
def compile_covariance(t, op, **kwargs):
    how = op.how

    fn = {"sample": F.covar_samp, "pop": F.covar_pop}[how]

    new_op = op.__class__(
        left=ops.Cast(op.left, to=dt.float64),
        right=ops.Cast(op.right, to=dt.float64),
        how=how,
        where=op.where,
    )
    return compile_aggregator(t, new_op, fn=fn, **kwargs)


@compiles(ops.Correlation)
def compile_correlation(t, op, **kwargs):
    if (how := op.how) == "pop":
        raise ValueError("PySpark only implements sample correlation")

    new_op = op.__class__(
        left=ops.Cast(op.left, to=dt.float64),
        right=ops.Cast(op.right, to=dt.float64),
        how=how,
        where=op.where,
    )
    return compile_aggregator(t, new_op, fn=F.corr, **kwargs)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, op, **kwargs):
    how = op.how

    if how == "first":
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == "last":
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise com.UnsupportedOperationError(
            f"PySpark backend does not support how={how!r}"
        )

    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.First)
def compile_first(t, op, **kwargs):
    fn = functools.partial(F.first, ignorenulls=True)
    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.Last)
def compile_last(t, op, **kwargs):
    fn = functools.partial(F.last, ignorenulls=True)
    return compile_aggregator(t, op, fn=fn, **kwargs)


@compiles(ops.Coalesce)
def compile_coalesce(t, op, **kwargs):
    kwargs["raw"] = False  # override to force column literals

    src_columns = []

    for arg in op.arg:
        col = t.translate(arg, **kwargs)

        if arg.dtype.is_floating():
            col = F.when(F.isnan(col), F.lit(None)).otherwise(col)

        src_columns.append(col)

    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.coalesce(*src_columns)


@compiles(ops.Greatest)
def compile_greatest(t, op, **kwargs):
    kwargs["raw"] = False  # override to force column literals
    src_columns = [t.translate(col, **kwargs) for col in op.arg]
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, op, **kwargs):
    kwargs["raw"] = False  # override to force column literals
    src_columns = [t.translate(col, **kwargs) for col in op.arg]
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.abs(src_column)


@compiles(ops.Clip)
def compile_clip(t, op, **kwargs):
    col = t.translate(op.arg, **kwargs)
    upper = t.translate(op.upper, **kwargs) if op.upper is not None else float("inf")
    lower = t.translate(op.lower, **kwargs) if op.lower is not None else float("-inf")

    def column_min(value, limit):
        """Return values greater than or equal to `limit`."""
        return F.when((value < limit) & ~F.isnull(value), limit).otherwise(value)

    def column_max(value, limit):
        """Return values less than or equal to `limit`."""
        return F.when((value > limit) & ~F.isnull(value), limit).otherwise(value)

    def clip(column, lower_value, upper_value):
        return column_max(column_min(column, F.lit(lower_value)), F.lit(upper_value))

    return clip(col, lower, upper).cast(PySparkType.from_ibis(op.dtype))


@compiles(ops.Round)
def compile_round(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    scale = t.translate(op.digits, **kwargs, raw=True) if op.digits is not None else 0
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype("long")
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)

    return F.when(src_column == 0, F.lit(0.0)).otherwise(
        F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0)
    )


@compiles(ops.Sqrt)
def compile_sqrt(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    raw_base = t.translate(op.base, **kwargs, raw=True)
    try:
        base = float(raw_base)
    except TypeError:
        return F.log(src_column) / F.log(raw_base)
    else:
        return F.log(base, src_column)


@compiles(ops.Ln)
def compile_ln(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    if op.dtype.is_boolean():
        return ~src_column
    return -src_column


@compiles(ops.Add)
def compile_add(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, op, **kwargs):
    arg = op.arg
    if arg.dtype.is_floating():
        src_column = t.translate(arg, **kwargs)
        return F.isnull(src_column) | F.isnan(src_column)
    return F.lit(False)


@compiles(ops.IsInf)
def compile_isinf(t, op, **kwargs):
    arg = op.arg
    if arg.dtype.is_floating():
        inf = float("inf")
        return t.translate(arg, **kwargs).isin([inf, -inf])
    return F.lit(False)


@compiles(ops.Uppercase)
def compile_uppercase(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.upper(src_column)


@compiles(ops.Lowercase)
def compile_lowercase(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.lower(src_column)


@compiles(ops.Reverse)
def compile_reverse(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.reverse(src_column)


@compiles(ops.Strip)
def compile_strip(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.trim(src_column)


@compiles(ops.LStrip)
def compile_lstrip(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.ltrim(src_column)


@compiles(ops.RStrip)
def compile_rstrip(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.rtrim(src_column)


@compiles(ops.Capitalize)
def compile_capitalize(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.initcap(src_column)


@compiles(ops.Substring)
def compile_substring(t, op, raw: bool = False, **kwargs):
    src_column = t.translate(op.arg, raw=raw, **kwargs)
    start = t.translate(op.start, **kwargs, raw=True) + 1
    length = t.translate(op.length, **kwargs, raw=True)

    if any_of((start, length), pyspark.sql.Column):
        raise NotImplementedError(
            "Specifying `start` or `length` with column expressions is not supported."
        )

    if start < 0:
        raise NotImplementedError("`start < 0` is not supported.")

    return src_column.substr(start, length)


@compiles(ops.StringLength)
def compile_string_length(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.length(src_column)


@compiles(ops.StrRight)
def compile_str_right(t, op, **kwargs):
    @F.udf("string")
    def str_right(s, nchars):
        return s[-nchars:]

    src_column = t.translate(op.arg, **kwargs)
    nchars_column = t.translate(op.nchars, **kwargs)
    return str_right(src_column, nchars_column)


@compiles(ops.Repeat)
def compile_repeat(t, op, **kwargs):
    @F.udf("string")
    def repeat(s, times):
        return s * times

    src_column = t.translate(op.arg, **kwargs)
    times_column = t.translate(op.times, **kwargs)
    return repeat(src_column, times_column)


@compiles(ops.StringFind)
def compile_string_find(t, op, **kwargs):
    @F.udf("long")
    def str_find(s, substr, start, end):
        return s.find(substr, start, end)

    src_column = t.translate(op.arg, **kwargs)
    substr_column = t.translate(op.substr, **kwargs)
    start_column = t.translate(op.start, **kwargs) if op.start else F.lit(None)
    end_column = t.translate(op.end, **kwargs) if op.end else F.lit(None)
    return str_find(src_column, substr_column, start_column, end_column)


@compiles(ops.Translate)
def compile_translate(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    from_str = op.from_str.value
    to_str = op.to_str.value
    return F.translate(src_column, from_str, to_str)


@compiles(ops.LPad)
def compile_lpad(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    length = op.length.value
    pad = op.pad.value
    return F.lpad(src_column, length, pad)


@compiles(ops.RPad)
def compile_rpad(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    length = op.length.value
    pad = op.pad.value
    return F.rpad(src_column, length, pad)


@compiles(ops.StringJoin)
def compile_string_join(t, op, **kwargs):
    @F.udf("string")
    def join(sep, arr):
        return sep.join(arr)

    sep_column = t.translate(op.sep, **kwargs)
    arg = [t.translate(arg, **kwargs) for arg in op.arg]
    return join(sep_column, F.array(arg))


@compiles(ops.RegexSearch)
def compile_regex_search(t, op, **kwargs):
    import re

    @F.udf("boolean")
    def regex_search(s, pattern):
        return re.search(pattern, s) is not None

    src_column = t.translate(op.arg, **kwargs)
    pattern = t.translate(op.pattern, **kwargs)
    return regex_search(src_column, pattern)


@compiles(ops.RegexExtract)
def compile_regex_extract(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    pattern = op.pattern.value
    idx = op.index.value
    return F.regexp_extract(src_column, pattern, idx)


@compiles(ops.RegexReplace)
def compile_regex_replace(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    pattern = op.pattern.value
    replacement = op.replacement.value
    return F.regexp_replace(src_column, pattern, replacement)


@compiles(ops.StringReplace)
def compile_string_replace(*args, **kwargs):
    return compile_regex_replace(*args, **kwargs)


@compiles(ops.StringSplit)
def compile_string_split(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    delimiter = op.delimiter.value
    return F.split(src_column, delimiter)


@compiles(ops.StringConcat)
def compile_string_concat(t, op, **kwargs):
    kwargs["raw"] = False  # override to force column literals
    src_columns = [t.translate(arg, **kwargs) for arg in op.arg]
    return F.concat(*src_columns)


@compiles(ops.StringAscii)
def compile_string_ascii(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.ascii(src_column)


@compiles(ops.StringSQLLike)
def compile_string_like(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    pattern = op.pattern.value
    return src_column.like(pattern)


@compiles(ops.InnerJoin)
def compile_inner_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="inner")


@compiles(ops.LeftJoin)
def compile_left_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="left")


@compiles(ops.RightJoin)
def compile_right_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="right")


@compiles(ops.OuterJoin)
def compile_outer_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="outer")


@compiles(ops.LeftSemiJoin)
def compile_left_semi_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="leftsemi")


@compiles(ops.LeftAntiJoin)
def compile_left_anti_join(t, op, **kwargs):
    return compile_join(t, op, **kwargs, how="leftanti")


def compile_join(t, op, how, **kwargs):
    left_df = t.translate(op.left, **kwargs)
    right_df = t.translate(op.right, **kwargs)

    pred_columns = reduce(
        operator.and_, map(partial(t.translate, **kwargs), op.predicates)
    )
    return left_df.join(right_df, pred_columns, how)


@compiles(ops.Distinct)
def compile_distinct(t, op, **kwargs):
    return t.translate(op.table, **kwargs).distinct()


def _canonicalize_interval(t, interval, **kwargs):
    """Convert interval to integer timestamp of second.

    When pyspark cast timestamp to integer type, it uses the number of
    seconds since epoch. Therefore, we need cast ibis interval
    correspondingly.
    """
    if isinstance(interval, ir.IntervalScalar):
        t.translate(interval.op(), **kwargs)
        return None

    elif isinstance(interval, int):
        return interval
    else:
        raise com.UnsupportedOperationError(
            f"type {type(interval)} is not supported in preceding /following "
            "in window."
        )


@compiles(ops.WindowBoundary)
def compile_window_boundary(t, boundary, **kwargs):
    if boundary.value.dtype.is_interval():
        value = t.translate(boundary.value, **kwargs)
        # TODO(kszucs): the value can be a literal which is a bug
        value = value.value if isinstance(value, ops.Literal) else value
        # value is in nanoseconds and spark uses seconds since epoch
        value = int(value / 1e9)
    else:
        value = boundary.value.value

    return -value if boundary.preceding else value


@compiles(ops.WindowFunction)
def compile_window_function(t, op, **kwargs):
    grouping_keys = [
        key.name if isinstance(key, ops.TableColumn) else t.translate(key, **kwargs)
        for key in op.frame.group_by
    ]

    # Timestamp needs to be cast to long for window bounds in spark
    ordering_keys = [
        F.col(sort.name).cast("long") if sort.dtype.is_timestamp() else sort.name
        for sort in op.frame.order_by
    ]
    aggcontext = AggregationContext.WINDOW
    pyspark_window = Window.partitionBy(grouping_keys).orderBy(ordering_keys)

    # If the operand is a shift op (e.g. lead, lag), Spark will set the window
    # bounds. Only set window bounds here if not a shift operation.
    func = op.func.__window_op__
    if not isinstance(func, ops.ShiftBase):
        if op.frame.start is None:
            win_start = Window.unboundedPreceding
        else:
            win_start = t.translate(op.frame.start, **kwargs)
        if op.frame.end is None:
            win_end = Window.unboundedFollowing
        else:
            win_end = t.translate(op.frame.end, **kwargs)

        if op.frame.how == "range":
            pyspark_window = pyspark_window.rangeBetween(win_start, win_end)
        else:
            pyspark_window = pyspark_window.rowsBetween(win_start, win_end)

    result = t.translate(func, **kwargs, aggcontext=aggcontext).over(pyspark_window)

    if isinstance(func, (ops.RankBase, ops.NTile)):
        # result must be cast to long type for Rank / RowNumber
        return result.astype("long") - 1
    else:
        return result


def _handle_shift_operation(t, op, fn, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    default = op.default.value if op.default is not None else op.default
    offset = op.offset.value if op.offset is not None else op.offset

    if offset:
        return fn(src_column, count=offset, default=default)
    else:
        return fn(src_column, default=default)


@compiles(ops.Lag)
def compile_lag(t, op, **kwargs):
    return _handle_shift_operation(t, op, fn=F.lag, **kwargs)


@compiles(ops.Lead)
def compile_lead(t, op, **kwargs):
    return _handle_shift_operation(t, op, fn=F.lead, **kwargs)


@compiles(ops.MinRank)
def compile_rank(t, op, **kwargs):
    return F.rank()


@compiles(ops.DenseRank)
def compile_dense_rank(t, op, **kwargs):
    return F.dense_rank()


@compiles(ops.PercentRank)
def compile_percent_rank(t, op, **kwargs):
    return F.percent_rank()


@compiles(ops.CumeDist)
def compile_cume_dist(t, op, **kwargs):
    raise com.UnsupportedOperationError(
        "PySpark backend does not support cume_dist with Ibis."
    )


@compiles(ops.NTile)
def compile_ntile(t, op, **kwargs):
    buckets = op.buckets.value
    return F.ntile(buckets)


@compiles(ops.FirstValue)
def compile_first_value(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.first(src_column)


@compiles(ops.LastValue)
def compile_last_value(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.last(src_column)


@compiles(ops.NthValue)
def compile_nth_value(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    nth = t.translate(op.nth, **kwargs, raw=True)
    return F.nth_value(src_column, nth + 1)


@compiles(ops.RowNumber)
def compile_row_number(t, op, **kwargs):
    return F.row_number()


# -------------------------- Temporal Operations ----------------------------

# Ibis value to PySpark value
_time_unit_mapping = {
    "Y": "year",
    "Q": "quarter",
    "M": "month",
    "W": "week",
    "D": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
}


@compiles(ops.Date)
def compile_date(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.to_date(src_column).cast("timestamp")


def _extract_component_from_datetime(t, op, extract_fn, **kwargs):
    date_col = t.translate(op.arg, **kwargs)
    return extract_fn(date_col).cast("integer")


@compiles(ops.ExtractYear)
def compile_extract_year(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.year, **kwargs)


@compiles(ops.ExtractMonth)
def compile_extract_month(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.month, **kwargs)


@compiles(ops.ExtractDay)
def compile_extract_day(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.dayofmonth, **kwargs)


@compiles(ops.ExtractDayOfYear)
def compile_extract_day_of_year(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.dayofyear, **kwargs)


@compiles(ops.ExtractQuarter)
def compile_extract_quarter(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.quarter, **kwargs)


@compiles(ops.ExtractEpochSeconds)
def compile_extract_epoch_seconds(t, op, **kwargs):
    return _extract_component_from_datetime(
        t, op, extract_fn=F.unix_timestamp, **kwargs
    )


@compiles(ops.ExtractWeekOfYear)
def compile_extract_week_of_year(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.weekofyear, **kwargs)


@compiles(ops.ExtractHour)
def compile_extract_hour(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.hour, **kwargs)


@compiles(ops.ExtractMinute)
def compile_extract_minute(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.minute, **kwargs)


@compiles(ops.ExtractSecond)
def compile_extract_second(t, op, **kwargs):
    return _extract_component_from_datetime(t, op, extract_fn=F.second, **kwargs)


@compiles(ops.ExtractMicrosecond)
def compile_extract_microsecond(t, op, **kwargs):
    raise com.UnsupportedOperationError(
        "PySpark backend does not support extracting microseconds."
    )


@compiles(ops.ExtractMillisecond)
def compile_extract_millisecond(t, op, **kwargs):
    raise com.UnsupportedOperationError(
        "PySpark backend does not support extracting milliseconds."
    )


@compiles(ops.DateTruncate)
def compile_date_truncate(t, op, **kwargs):
    try:
        unit = _time_unit_mapping[op.unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(
            f"{op.unit!r} unit is not supported in timestamp truncate"
        )

    src_column = t.translate(op.arg, **kwargs)
    return F.date_trunc(unit, src_column)


@compiles(ops.TimestampTruncate)
def compile_timestamp_truncate(t, op, **kwargs):
    return compile_date_truncate(t, op, **kwargs)


@compiles(ops.Strftime)
def compile_strftime(t, op, **kwargs):
    format_str = op.format_str.value

    @pandas_udf("string", PandasUDFType.SCALAR)
    def strftime(timestamps):
        return timestamps.dt.strftime(format_str)

    src_column = t.translate(op.arg, **kwargs)
    return strftime(src_column)


@compiles(ops.TimestampFromUNIX)
def compile_timestamp_from_unix(t, op, **kwargs):
    unixtime = t.translate(op.arg, **kwargs)
    if not op.unit:
        return F.to_timestamp(F.from_unixtime(unixtime))
    elif op.unit.short == "s":
        fmt = "yyyy-MM-dd HH:mm:ss"
        return F.to_timestamp(F.from_unixtime(unixtime, fmt), fmt)
    else:
        raise com.UnsupportedArgumentError(
            "PySpark backend does not support timestamp from unix time with "
            f"unit {op.unit.short}. Supported unit is s."
        )


@compiles(ops.TimestampNow)
def compile_timestamp_now(t, op, **kwargs):
    return F.current_timestamp()


@compiles(ops.StringToTimestamp)
def compile_string_to_timestamp(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    fmt = op.format_str.value
    return F.to_timestamp(src_column, fmt)


@compiles(ops.DayOfWeekIndex)
def compile_day_of_week_index(t, op, **kwargs):
    @pandas_udf("short", PandasUDFType.SCALAR)
    def day_of_week(s):
        return s.dt.dayofweek

    src_column = t.translate(op.arg, **kwargs)
    return day_of_week(src_column.cast("timestamp"))


@compiles(ops.DayOfWeekName)
def compiles_day_of_week_name(t, op, **kwargs):
    @pandas_udf("string", PandasUDFType.SCALAR)
    def day_name(s):
        return s.dt.day_name()

    src_column = t.translate(op.arg, **kwargs)
    return day_name(src_column.cast("timestamp"))


def _get_interval_col(t, op, allowed_units=None, **kwargs):
    import pandas as pd

    dtype = op.dtype
    if not dtype.is_interval():
        raise com.UnsupportedArgumentError(
            f"{dtype} expression cannot be converted to interval column. "
            "Must be Interval dtype."
        )
    if allowed_units and dtype.unit.short not in allowed_units:
        raise com.UnsupportedArgumentError(
            f'Interval unit "{dtype.unit.short}" is not allowed. Allowed units are: '
            f"{allowed_units}"
        )

    # if interval expression is a binary op, translate expression into
    # an interval column and return
    if isinstance(op, ops.IntervalBinary):
        return t.translate(op, **kwargs)

    # otherwise, translate expression into a literal op and construct
    # interval column from literal value and dtype
    if isinstance(op, ops.Alias):
        op = op.arg

    # TODO(kszucs): t.translate should never return with an ibis operation;
    # I assume this is required for special case when casting to intervals,
    # see the implementation of ops.Cast compilation
    if not isinstance(op, ops.Literal):
        op = t.translate(op, **kwargs)

    if isinstance(op.value, pd.Timedelta):
        td_nanos = op.value.value
        if td_nanos % 1000 != 0:
            raise com.UnsupportedArgumentError(
                "Interval with nanoseconds is not supported. The "
                "smallest unit supported by Spark is microseconds."
            )
        td_micros = td_nanos // 1000
        return F.expr(f"INTERVAL {td_micros} MICROSECOND")
    else:
        return F.expr(f"INTERVAL {op.value} {_time_unit_mapping[dtype.unit.short]}")


def _compile_datetime_binop(t, op, *, fn, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = _get_interval_col(t, op.right, **kwargs)
    return fn(left, right)


@compiles(ops.DateAdd)
def compile_date_add(t, op, **kwargs):
    allowed_units = ["Y", "W", "M", "D"]
    return _compile_datetime_binop(
        t,
        op,
        fn=lambda lhs, rhs: (lhs + rhs).cast("timestamp"),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.DateSub)
def compile_date_sub(t, op, **kwargs):
    allowed_units = ["Y", "W", "M", "D"]
    return _compile_datetime_binop(
        t,
        op,
        fn=lambda lhs, rhs: (lhs - rhs).cast("timestamp"),
        allowed_units=allowed_units,
        **kwargs,
    )


if vparse(pyspark.__version__) >= vparse("3.3"):

    @compiles(ops.DateDiff)
    def compile_date_diff(t, op, **kwargs):
        left = t.translate(op.left, **kwargs)
        right = t.translate(op.right, **kwargs)

        return F.concat(
            F.lit("INTERVAL '"), F.datediff(left, right), F.lit("' DAY")
        ).cast(
            pt.DayTimeIntervalType(
                pt.DayTimeIntervalType.DAY, pt.DayTimeIntervalType.DAY
            )
        )


@compiles(ops.TimestampAdd)
def compile_timestamp_add(t, op, **kwargs):
    allowed_units = ["Y", "W", "M", "D", "h", "m", "s"]
    return _compile_datetime_binop(
        t,
        op,
        fn=lambda lhs, rhs: (lhs + rhs).cast("timestamp"),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampSub)
def compile_timestamp_sub(t, op, **kwargs):
    allowed_units = ["Y", "W", "M", "D", "h", "m", "s"]
    return _compile_datetime_binop(
        t,
        op,
        fn=lambda lhs, rhs: (lhs - rhs).cast("timestamp"),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampDiff)
def compile_timestamp_diff(t, op, **kwargs):
    raise com.UnsupportedOperationError(
        "PySpark backend does not support TimestampDiff as there is no "
        "timedelta type."
    )


def _compile_interval_binop(t, op, fn, **kwargs):
    left = _get_interval_col(t, op.left, **kwargs)
    right = _get_interval_col(t, op.right, **kwargs)

    return fn(left, right)


@compiles(ops.IntervalAdd)
def compile_interval_add(t, op, **kwargs):
    return _compile_interval_binop(t, op, fn=operator.add, **kwargs)


@compiles(ops.IntervalSubtract)
def compile_interval_subtract(t, op, **kwargs):
    return _compile_interval_binop(t, op, fn=operator.sub, **kwargs)


@compiles(ops.IntervalFromInteger)
def compile_interval_from_integer(t, op, **kwargs):
    raise com.UnsupportedOperationError(
        "Interval from integer column is unsupported for the PySpark backend."
    )


# -------------------------- Array Operations ----------------------------


@compiles(ops.ArrayColumn)
def compile_array_column(t, op, **kwargs):
    cols = [t.translate(col, **kwargs) for col in op.cols]
    return F.array(cols)


@compiles(ops.ArrayLength)
def compile_array_length(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.size(src_column)


@compiles(ops.ArraySlice)
def compile_array_slice(t, op, **kwargs):
    start = op.start.value if op.start is not None else op.start
    stop = op.stop.value if op.stop is not None else op.stop
    spark_type = PySparkType.from_ibis(op.arg.dtype)

    @F.udf(spark_type)
    def array_slice(array):
        return array[start:stop]

    src_column = t.translate(op.arg, **kwargs)
    return array_slice(src_column)


@compiles(ops.ArrayIndex)
def compile_array_index(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    index = op.index.value + 1
    return F.element_at(src_column, index)


@compiles(ops.ArrayConcat)
def compile_array_concat(t, op, **kwargs):
    return F.concat(*map(partial(t.translate, **kwargs), op.arg))


@compiles(ops.ArrayRepeat)
def compile_array_repeat(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    times = op.times.value
    return F.flatten(F.array_repeat(src_column, times))


@compiles(ops.ArrayCollect)
def compile_array_collect(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    if (where := op.where) is not None:
        src_column = F.when(t.translate(where, **kwargs), src_column)
    return F.collect_list(src_column)


@compiles(ops.Argument)
def compile_argument(t, op, arg_columns, **kwargs):
    return arg_columns[op.name]


@compiles(ops.ArrayFilter)
def compile_array_filter(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.filter(
        src_column,
        lambda x: t.translate(op.body, arg_columns={op.param: x}, **kwargs),
    )


@compiles(ops.ArrayMap)
def compile_array_map(t, op, **kwargs):
    src_column = t.translate(op.arg, **kwargs)
    return F.transform(
        src_column,
        lambda x: t.translate(op.body, arg_columns={op.param: x}, **kwargs),
    )


# --------------------------- Null Operations -----------------------------


@compiles(ops.NullIf)
def compile_null_if(t, op, **kwargs):
    col = t.translate(op.arg, **kwargs)
    nullif_col = t.translate(op.null_if_expr, **kwargs)
    return F.when(col == nullif_col, F.lit(None)).otherwise(col)


@compiles(ops.IsNull)
def compile_is_null(t, op, **kwargs):
    arg = op.arg
    col = t.translate(arg, **kwargs)
    result = F.isnull(col)
    if arg.dtype.is_floating():
        result |= F.isnan(col)
    return result


@compiles(ops.NotNull)
def compile_not_null(t, op, **kwargs):
    arg = op.arg
    col = t.translate(arg, **kwargs)
    result = ~F.isnull(col)
    if arg.dtype.is_floating():
        result &= ~F.isnan(col)
    return result


@compiles(ops.DropNa)
def compile_dropna_table(t, op, **kwargs):
    table = t.translate(op.table, **kwargs)

    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None

    return table.dropna(how=op.how, subset=subset)


@compiles(ops.FillNa)
def compile_fillna_table(t, op, **kwargs):
    table = t.translate(op.table, **kwargs)
    raw_replacements = op.replacements
    replacements = (
        dict(raw_replacements)
        if isinstance(raw_replacements, frozendict)
        else raw_replacements.value
    )
    return table.fillna(replacements)


# ------------------------- User defined function ------------------------


@compiles(ops.ElementWiseVectorizedUDF)
def compile_elementwise_udf(t, op, **kwargs):
    spark_output_type = PySparkType.from_ibis(op.return_type)
    func = op.func
    spark_udf = pandas_udf(func, spark_output_type, PandasUDFType.SCALAR)
    func_args = (t.translate(arg, **kwargs) for arg in op.func_args)
    return spark_udf(*func_args)


@compiles(ops.ReductionVectorizedUDF)
def compile_reduction_udf(t, op, *, aggcontext=None, **kwargs):
    spark_output_type = PySparkType.from_ibis(op.return_type)
    spark_udf = pandas_udf(op.func, spark_output_type, PandasUDFType.GROUPED_AGG)
    func_args = (t.translate(arg, **kwargs) for arg in op.func_args)

    col = spark_udf(*func_args)
    if aggcontext:
        return col
    else:
        src_table = t.translate(op.func_args[0].table, **kwargs)
        return src_table.agg(col)


# NB: this is intentionally not using @compiles because @compiles doesn't
# handle subclasses of operations
def compile_scalar_udf(t, op, **kwargs):
    if op.__input_type__ != InputType.PANDAS:
        raise NotImplementedError("Only Pandas UDFs are support in the PySpark backend")

    import pandas as pd

    make_series = partial(pd.Series, dtype=op.dtype.to_pandas())
    func = toolz.compose(make_series, op.__func__)
    spark_dtype = PySparkType.from_ibis(op.dtype)
    spark_udf = pandas_udf(func, spark_dtype, PandasUDFType.SCALAR)
    return spark_udf(*map(partial(t.translate, **kwargs), op.args))


@compiles(ops.SearchedCase)
def compile_searched_case(t, op, **kwargs):
    existing_when = None

    for case, result in zip(op.cases, op.results):
        if existing_when is not None:
            # Spark allowed chained when statement
            when = existing_when.when
        else:
            when = F.when

        existing_when = when(
            t.translate(case, **kwargs),
            t.translate(result, **kwargs),
        )

    return existing_when.otherwise(t.translate(op.default, **kwargs))


@compiles(ops.View)
def compile_view(t, op, **kwargs):
    name = op.name
    child = op.child
    # TODO(kszucs): avoid converting to expr
    backend = child.to_expr()._find_backend()
    tables = backend._session.catalog.listTables()
    if any(name == table.name and not table.isTemporary for table in tables):
        raise ValueError(f"table or non-temporary view `{name}` already exists")
    result = t.translate(child, **kwargs)
    result.createOrReplaceTempView(name)
    return result.alias(name)


@compiles(ops.SQLStringView)
def compile_sql_view(t, op, **kwargs):
    # TODO(kszucs): avoid converting to expr
    backend = op.child.to_expr()._find_backend()
    result = backend._session.sql(op.query)
    name = op.name
    result.createOrReplaceTempView(name)
    return result.alias(name)


@compiles(ops.StringContains)
def compile_string_contains(t, op, **kwargs):
    haystack = t.translate(op.haystack, **kwargs)
    needle = t.translate(op.needle, **kwargs)
    return haystack.contains(needle)


@compiles(ops.Unnest)
def compile_unnest(t, op, **kwargs):
    column = t.translate(op.arg, **kwargs)
    return F.explode(column)


@compiles(ops.Acos)
@compiles(ops.Asin)
@compiles(ops.Atan)
@compiles(ops.Cos)
@compiles(ops.Sin)
@compiles(ops.Tan)
def compile_trig(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    func_name = op.__class__.__name__.lower()
    func = getattr(F, func_name)
    return func(arg)


@compiles(ops.Cot)
def compile_cot(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return 1.0 / F.tan(arg)


@compiles(ops.Atan2)
def compile_atan2(t, op, **kwargs):
    y, x = (t.translate(arg, **kwargs) for arg in op.args)
    return F.atan2(y, x)


@compiles(ops.Degrees)
def compile_degrees(t, op, **kwargs):
    return F.degrees(t.translate(op.arg, **kwargs))


@compiles(ops.Radians)
def compile_radians(t, op, **kwargs):
    return F.radians(t.translate(op.arg, **kwargs))


@compiles(ops.IfElse)
def compile_ifelse(t, op, **kwargs):
    return F.when(
        t.translate(op.bool_expr, **kwargs),
        t.translate(op.true_expr, **kwargs),
    ).otherwise(t.translate(op.false_null_expr, **kwargs))


@compiles(ops.RandomScalar)
def compile_random(*args, **kwargs):
    return F.rand()


@compiles(ops.InMemoryTable)
def compile_in_memory_table(t, op, session, **kwargs):
    fields = [
        pt.StructField(name, PySparkType.from_ibis(dtype), dtype.nullable)
        for name, dtype in op.schema.items()
    ]
    schema = pt.StructType(fields)
    return session.createDataFrame(data=op.data.to_frame(), schema=schema).alias(
        op.name
    )


@compiles(ops.BitwiseAnd)
def compile_bitwise_and(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)

    return left.bitwiseAND(right)


@compiles(ops.BitwiseOr)
def compile_bitwise_or(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)

    return left.bitwiseOR(right)


@compiles(ops.BitwiseXor)
def compile_bitwise_xor(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)

    return left.bitwiseXOR(right)


@compiles(ops.BitwiseNot)
def compile_bitwise_not(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return F.bitwise_not(arg)


@compiles(ops.JSONGetItem)
def compile_json_getitem(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    index = t.translate(op.index, raw=True, **kwargs)
    if op.index.dtype.is_integer():
        path = f"$[{index}]"
    else:
        path = f"$.{index}"
    return F.get_json_object(arg, path)


@compiles(ops.DummyTable)
def compile_dummy_table(t, op, session=None, **kwargs):
    return session.range(0, 1).select(
        *(t.translate(value, **kwargs) for value in op.values)
    )


@compiles(ops.ScalarParameter)
def compile_scalar_parameter(t, op, timecontext=None, scope=None, **kwargs):
    assert scope is not None, "scope is None"
    raw_value = scope.get_value(op, timecontext)
    return F.lit(raw_value).cast(PySparkType.from_ibis(op.dtype))


@compiles(ops.E)
def compile_e(t, op, **kwargs):
    return F.exp(F.lit(1))


@compiles(ops.Pi)
def compile_pi(t, op, **kwargs):
    return F.acos(F.lit(-1))


@compiles(ops.Quantile)
@compiles(ops.MultiQuantile)
def compile_quantile(t, op, **kwargs):
    return compile_aggregator(
        t, op, fn=F.percentile_approx, where_excludes=("quantile",), **kwargs
    )


@compiles(ops.ArgMin)
def compile_argmin(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.min_by, **kwargs)


@compiles(ops.ArgMax)
def compile_argmax(t, op, **kwargs):
    return compile_aggregator(t, op, fn=F.max_by, **kwargs)


@compiles(ops.ArrayStringJoin)
def compile_array_string_join(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    sep = t.translate(op.sep, raw=True, **kwargs)
    return F.concat_ws(sep, arg)


@compiles(ops.ArrayContains)
def compile_array_contains(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    other = t.translate(op.other, **kwargs)
    return F.when(
        ~F.isnull(arg), F.coalesce(F.array_contains(arg, other), F.lit(False))
    ).otherwise(F.lit(None))


@compiles(ops.ArrayPosition)
def compile_array_pos(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    other = t.translate(op.other, raw=True, **kwargs)
    return F.array_position(arg, other) - 1


@compiles(ops.ArrayDistinct)
def compile_array_distinct(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return F.array_distinct(arg)


@compiles(ops.ArraySort)
def compile_array_sort(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    return F.array_sort(arg)


@compiles(ops.ArrayRemove)
def compile_array_remove(t, op, **kwargs):
    arg = t.translate(op.arg, **kwargs)
    other = t.translate(op.other, raw=True, **kwargs)
    return F.array_remove(arg, other)


@compiles(ops.ArrayUnion)
def compile_array_union(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return F.array_union(left, right)


@compiles(ops.ArrayIntersect)
def compile_array_intersect(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return F.array_intersect(left, right)


@compiles(ops.Hash)
def compile_hash_column(t, op, **kwargs):
    return F.hash(t.translate(op.arg, **kwargs))


@compiles(ops.ArrayZip)
def compile_zip(t, op, **kwargs):
    return F.arrays_zip(*map(partial(t.translate, **kwargs), op.arg))


@compiles(ops.Levenshtein)
def compile_levenshtein(t, op, **kwargs):
    left = t.translate(op.left, **kwargs)
    right = t.translate(op.right, **kwargs)
    return F.levenshtein(left, right)
