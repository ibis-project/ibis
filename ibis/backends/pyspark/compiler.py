import collections
import enum
import functools
import operator

import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as pt
from pyspark.sql import Window
from pyspark.sql.functions import PandasUDFType, pandas_udf

import ibis.common.exceptions as com
import ibis.expr.datatypes as dtypes
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.types as types
from ibis import interval
from ibis.backends.pandas.client import PandasInMemoryTable
from ibis.backends.pandas.execution import execute
from ibis.backends.pyspark.datatypes import (
    ibis_array_dtype_to_spark_dtype,
    ibis_dtype_to_spark_dtype,
    spark_dtype,
)
from ibis.backends.pyspark.timecontext import (
    combine_time_context,
    filter_by_time_context,
)
from ibis.config import options
from ibis.expr.timecontext import adjust_context
from ibis.util import frozendict, guid


class PySparkDatabaseTable(ops.DatabaseTable):
    pass


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

    def translate(self, expr, scope, timecontext, **kwargs):
        """
        Translate Ibis expression into a PySpark object.

        All translated expressions are cached within scope. If an expression is
        found within scope, it's returned. Otherwise, the it's translated and
        cached for future reference.

        :param expr: ibis expression
        :param scope: dictionary mapping from operation to translated result
        :param timecontext: time context associated with expr
        :param kwargs: parameters passed as keyword args (e.g. window)
        :return: translated PySpark DataFrame or Column object
        """
        # The operation node type the typed expression wraps
        op = expr.op()

        result = scope.get_value(op, timecontext)
        if result is not None:
            return result
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            result = formatter(self, expr, scope, timecontext, **kwargs)
            scope.set_value(op, timecontext, result)
            return result
        else:
            raise com.OperationNotDefinedError(
                f'No translation rule for {type(op)}'
            )


compiles = PySparkExprTranslator.compiles


@compiles(PySparkDatabaseTable)
def compile_datasource(t, expr, scope, timecontext, **_):
    op = expr.op()
    name, _, client = op.args
    return filter_by_time_context(client._session.table(name), timecontext)


@compiles(ops.SQLQueryResult)
def compile_sql_query_result(t, expr, scope, timecontext, **_):
    op = expr.op()
    query, _, client = op.args
    return client._session.sql(query)


def _can_be_replaced_by_column_name(column_expr, table):
    """
    Return whether the given column_expr can be replaced by its literal
    name, which is True when column_expr and table[column_expr.get_name()]
    is semantically the same.
    """
    # Each check below is necessary to distinguish a pure projection from
    # other valid selections, such as a mutation that assigns a new column
    # or changes the value of an existing column.
    return (
        isinstance(column_expr.op(), ops.TableColumn)
        and column_expr.op().table == table
        and column_expr.get_name() in table.schema()
        and column_expr.op() == table[column_expr.get_name()].op()
    )


@compiles(ops.Alias)
def compile_alias(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    return arg.alias(op.name)


@compiles(ops.Selection)
def compile_selection(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    # In selection, there could be multiple children that point to the
    # same root table. e.g. window with different sizes on a table.
    # We need to get the 'combined' time range that is a superset of every
    # time context among child nodes, and pass this as context to
    # source table to get all data within time context loaded.
    arg_timecontexts = [
        adjust_context(node.op(), scope, timecontext)
        for node in op.selections
        if timecontext
    ]
    adjusted_timecontext = combine_time_context(arg_timecontexts)
    # If this is a sort or filter node, op.selections is empty
    # in this case, we use the original timecontext
    if not adjusted_timecontext:
        adjusted_timecontext = timecontext
    src_table = t.translate(op.table, scope, adjusted_timecontext, **kwargs)

    col_in_selection_order = []
    col_to_drop = []
    result_table = src_table

    for predicate in op.predicates:
        col = t.translate(predicate, scope, timecontext, **kwargs)
        # Due to an upstream Spark issue (SPARK-33057) we cannot
        # directly use filter with a window operation. The workaround
        # here is to assign a temporary column for the filter predicate,
        # do the filtering, and then drop the temporary column.
        filter_column = f'predicate_{guid()}'
        result_table = result_table.withColumn(filter_column, col)
        result_table = result_table.filter(F.col(filter_column))
        result_table = result_table.drop(filter_column)

    for selection in op.selections:
        if isinstance(selection, types.Table):
            col_in_selection_order.extend(selection.columns)
        elif isinstance(selection, types.DestructColumn):
            struct_col = t.translate(
                selection, scope, adjusted_timecontext, **kwargs
            )
            # assign struct col and drop it later
            # This is a work around to ensure that the struct_col
            # is only executed once
            struct_col_name = f"destruct_col_{guid()}"
            result_table = result_table.withColumn(struct_col_name, struct_col)
            col_to_drop.append(struct_col_name)
            cols = [
                result_table[struct_col_name][name].alias(name)
                for name in selection.type().names
            ]
            col_in_selection_order.extend(cols)
        elif isinstance(selection, (types.Column, types.Scalar)):
            # If the selection is a straightforward projection of a table
            # column from the root table itself (i.e. excluding mutations and
            # renames), we can get the selection name directly.
            if _can_be_replaced_by_column_name(selection, op.table):
                col_in_selection_order.append(selection.get_name())
            else:
                col = t.translate(
                    selection, scope, adjusted_timecontext, **kwargs
                ).alias(selection.get_name())
                col_in_selection_order.append(col)
        else:
            raise NotImplementedError(
                f"Unrecoginized type in selections: {type(selection)}"
            )
    if col_in_selection_order:
        result_table = result_table[col_in_selection_order]

    if col_to_drop:
        result_table = result_table.drop(*col_to_drop)

    if op.sort_keys:
        sort_cols = [
            t.translate(key, scope, timecontext, **kwargs)
            for key in op.sort_keys
        ]
        result_table = result_table.sort(*sort_cols)

    return filter_by_time_context(
        result_table, timecontext, adjusted_timecontext
    )


@compiles(ops.SortKey)
def compile_sort_key(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.expr, scope, timecontext, **kwargs)

    if op.ascending:
        return col.asc()
    else:
        return col.desc()


def compile_nan_as_null(compile_func):
    @functools.wraps(compile_func)
    def wrapper(t, expr, *args, **kwargs):
        compiled = compile_func(t, expr, *args, **kwargs)
        if options.pyspark.treat_nan_as_null and isinstance(
            expr.type(), dtypes.Floating
        ):
            return F.nanvl(compiled, F.lit(None))
        else:
            return compiled

    return wrapper


@compiles(ops.TableColumn)
@compile_nan_as_null
def compile_column(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    table = t.translate(op.table, scope, timecontext, **kwargs)
    return table[op.name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.table, scope, timecontext, **kwargs)


@compiles(ops.Cast)
def compile_cast(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    if isinstance(op.to, dtypes.Interval):
        if isinstance(op.arg.op(), ops.Literal):
            return interval(op.arg.op().value, op.to.unit)
        else:
            raise com.UnsupportedArgumentError(
                'Casting to intervals is only supported for literals '
                'in the PySpark backend. {} not allowed.'.format(type(op.arg))
            )

    if isinstance(op.to, dtypes.Array):
        cast_type = ibis_array_dtype_to_spark_dtype(op.to)
    else:
        cast_type = ibis_dtype_to_spark_dtype(op.to)

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return src_column.cast(cast_type)


@compiles(ops.Limit)
def compile_limit(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    if op.offset != 0:
        raise com.UnsupportedArgumentError(
            'PySpark backend does not support non-zero offset is for '
            'limit operation. Got offset {}.'.format(op.offset)
        )
    df = t.translate(op.table, scope, timecontext, **kwargs)
    return df.limit(op.n)


@compiles(ops.And)
def compile_and(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) & t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Or)
def compile_or(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) | t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Xor)
def compile_xor(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return (left | right) & ~(left & right)


@compiles(ops.Equals)
def compile_equals(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) == t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Not)
def compile_not(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return ~t.translate(op.arg, scope, timecontext, **kwargs)


@compiles(ops.NotEquals)
def compile_not_equals(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) != t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Greater)
def compile_greater(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) > t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) >= t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Less)
def compile_less(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) < t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.LessEqual)
def compile_less_equal(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) <= t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Multiply)
def compile_multiply(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) * t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Subtract)
def compile_subtract(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope, timecontext, **kwargs) - t.translate(
        op.right, scope, timecontext, **kwargs
    )


@compiles(ops.Literal)
@compile_nan_as_null
def compile_literal(t, expr, scope, timecontext, raw=False, **kwargs):
    """If raw is True, don't wrap the result with F.lit()"""
    value = expr.op().value
    dtype = expr.op().dtype

    if raw:
        return value

    if isinstance(dtype, dtypes.Interval):
        # execute returns a Timedelta and value is nanoseconds
        return execute(expr).value

    if isinstance(value, collections.abc.Set):
        # Don't wrap set with F.lit
        if isinstance(value, frozenset):
            # Spark doens't like frozenset
            return set(value)
        else:
            return value
    elif isinstance(value, tuple):
        return F.array(*map(F.lit, value))
    else:
        if isinstance(value, pd.Timestamp) and value.tz is None:
            value = value.tz_localize("UTC").to_pydatetime()
        return F.lit(value)


def _compile_agg(t, agg_expr, scope, timecontext, *, context, **kwargs):
    agg = t.translate(agg_expr, scope, timecontext, context=context, **kwargs)
    if agg_expr.has_name():
        return agg.alias(agg_expr.get_name())
    return agg


@compiles(ops.Aggregation)
def compile_aggregation(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_table = t.translate(op.table, scope, timecontext, **kwargs)

    if op.predicates:
        src_table = src_table.filter(
            t.translate(
                functools.reduce(operator.and_, op.predicates),
                scope,
                timecontext,
                **kwargs,
            )
        )

    if op.by:
        context = AggregationContext.GROUP
        bys = [t.translate(b, scope, timecontext, **kwargs) for b in op.by]
        src_table = src_table.groupby(*bys)
    else:
        context = AggregationContext.ENTIRE

    aggs = [
        _compile_agg(t, m, scope, timecontext, context=context)
        for m in op.metrics
    ]
    return src_table.agg(*aggs)


@compiles(ops.Union)
def compile_union(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    result = left.union(right)
    return result.distinct() if op.distinct else result


@compiles(ops.Intersection)
def compile_intersection(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return left.intersect(right) if op.distinct else left.intersectAll(right)


@compiles(ops.Difference)
def compile_difference(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return left.subtract(right) if op.distinct else left.exceptAll(right)


@compiles(ops.Contains)
def compile_contains(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.value, scope, timecontext, **kwargs)
    return col.isin(t.translate(op.options, scope, timecontext, **kwargs))


@compiles(ops.NotContains)
def compile_not_contains(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.value, scope, timecontext, **kwargs)
    return ~(col.isin(t.translate(op.options, scope, timecontext, **kwargs)))


@compiles(ops.StartsWith)
def compile_startswith(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    start = t.translate(op.start, scope, timecontext, **kwargs)
    return col.startswith(start)


@compiles(ops.EndsWith)
def compile_endswith(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    end = t.translate(op.end, scope, timecontext, **kwargs)
    return col.startswith(end)


def _is_table(table):
    try:
        return isinstance(table.op().arg, ir.Table)
    except AttributeError:
        return False


def compile_aggregator(
    t, expr, scope, timecontext, *, fn, context=None, **kwargs
):
    op = expr.op()
    if (where := getattr(op, 'where', None)) is not None:
        condition = t.translate(where, scope, timecontext, **kwargs)
    else:
        condition = None

    def translate_arg(arg):
        src_col = t.translate(arg, scope, timecontext, **kwargs)

        if condition is not None:
            src_col = F.when(condition, src_col)
        return src_col

    src_inputs = tuple(
        arg for arg in op.args if arg is not getattr(op, "where", None)
    )
    src_cols = tuple(
        translate_arg(arg) for arg in src_inputs if isinstance(arg, ir.Expr)
    )

    col = fn(*src_cols)
    if context:
        return col
    else:
        # We are trying to compile a expr such as some_col.max()
        # to a Spark expression.
        # Here we get the root table df of that column and compile
        # the expr to:
        # df.select(max(some_col))
        if _is_table(expr):
            (src_col,) = src_cols
            return src_col.select(col)
        (table_op,) = op.root_tables()
        return t.translate(
            table_op.to_expr(), scope, timecontext, **kwargs
        ).select(col)


@compiles(ops.GroupConcat)
def compile_group_concat(t, expr, scope, timecontext, context=None, **kwargs):
    sep = t.translate(expr.op().sep, scope, timecontext, raw=True)

    def fn(col, _):
        collected = F.collect_list(col)
        return F.array_join(
            F.when(F.size(collected) == 0, F.lit(None)).otherwise(collected),
            sep,
        )

    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context
    )


@compiles(ops.Any)
def compile_any(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.max, context=context, **kwargs
    )


@compiles(ops.NotAny)
def compile_notany(t, expr, scope, timecontext, *, context=None, **kwargs):
    # The code here is a little ugly because the translation are different
    # with different context.
    # When translating col.notany() (context is None), we returns the dataframe
    # so we need to negate the aggregator, i.e., df.select(~F.max(col))
    # When traslating col.notany().over(w), we need to negate the result
    # after the window translation, i.e., ~(F.max(col).over(w))

    if context is None:

        def fn(col):
            return ~(F.max(col))

        return compile_aggregator(
            t, expr, scope, timecontext, fn=fn, context=context, **kwargs
        )
    else:
        return ~compile_any(
            t,
            expr,
            scope,
            timecontext,
            context=context,
            **kwargs,
        )


@compiles(ops.All)
def compile_all(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.min, context=context, **kwargs
    )


@compiles(ops.NotAll)
def compile_notall(t, expr, scope, timecontext, *, context=None, **kwargs):
    # See comments for opts.NotAny for reasoning for the if/else
    if context is None:

        def fn(col):
            return ~(F.min(col))

        return compile_aggregator(
            t, expr, scope, timecontext, fn=fn, context=context, **kwargs
        )
    else:
        return ~compile_all(
            t,
            expr,
            scope,
            timecontext,
            context=context,
            **kwargs,
        )


def _count_star(_):
    return F.count(F.lit(1))


@compiles(ops.Count)
def compile_count(t, expr, scope, timecontext, context=None, **kwargs):
    if _is_table(expr):
        fn = _count_star
    else:
        fn = F.count
    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context, **kwargs
    )


@compiles(ops.Max)
@compiles(ops.CumulativeMax)
def compile_max(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.max, context=context, **kwargs
    )


@compiles(ops.Min)
@compiles(ops.CumulativeMin)
def compile_min(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.min, context=context, **kwargs
    )


@compiles(ops.Mean)
@compiles(ops.CumulativeMean)
def compile_mean(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.mean, context=context, **kwargs
    )


@compiles(ops.Sum)
@compiles(ops.CumulativeSum)
def compile_sum(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.sum, context=context, **kwargs
    )


@compiles(ops.ApproxCountDistinct)
def compile_approx_count_distinct(
    t, expr, scope, timecontext, context=None, **kwargs
):
    return compile_aggregator(
        t,
        expr,
        scope,
        timecontext,
        fn=F.approx_count_distinct,
        context=context,
        **kwargs,
    )


@compiles(ops.ApproxMedian)
def compile_approx_median(t, expr, scope, timecontext, context=None, **kwargs):
    return compile_aggregator(
        t,
        expr,
        scope,
        timecontext,
        fn=lambda arg: F.percentile_approx(arg, 0.5),
        context=context,
        **kwargs,
    )


@compiles(ops.StandardDev)
def compile_std(t, expr, scope, timecontext, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise com.TranslationError(f"Unexpected 'how' in translation: {how}")

    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context
    )


@compiles(ops.Variance)
def compile_variance(t, expr, scope, timecontext, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise com.TranslationError(f"Unexpected 'how' in translation: {how}")

    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context
    )


@compiles(ops.Covariance)
def compile_covariance(t, expr, scope, timecontext, context=None, **kwargs):
    op = expr.op()
    how = op.how

    fn = {"sample": F.covar_samp, "pop": F.covar_pop}[how]

    pyspark_double_type = ibis_dtype_to_spark_dtype(dtypes.double)
    expr = op.__class__(
        left=op.left.cast(pyspark_double_type),
        right=op.right.cast(pyspark_double_type),
        how=how,
        where=op.where,
    ).to_expr()
    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context
    )


@compiles(ops.Correlation)
def compile_correlation(t, expr, scope, timecontext, context=None, **kwargs):
    op = expr.op()

    if (how := op.how) == "pop":
        raise ValueError("PySpark only implements sample correlation")

    pyspark_double_type = ibis_dtype_to_spark_dtype(dtypes.double)
    expr = op.__class__(
        left=op.left.cast(pyspark_double_type),
        right=op.right.cast(pyspark_double_type),
        how=how,
        where=op.where,
    ).to_expr()
    return compile_aggregator(
        t, expr, scope, timecontext, fn=F.corr, context=context
    )


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, scope, timecontext, context=None, **kwargs):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError(f"Does not support 'how': {how}")

    return compile_aggregator(
        t, expr, scope, timecontext, fn=fn, context=context
    )


@compiles(ops.Coalesce)
def compile_coalesce(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope, timecontext, **kwargs)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.coalesce(*src_columns)


@compiles(ops.Greatest)
def compile_greatest(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope, timecontext, **kwargs)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope, timecontext, **kwargs)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.abs(src_column)


@compiles(ops.Clip)
def compile_clip(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    spark_dtype = ibis_dtype_to_spark_dtype(expr.type())
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    upper = (
        t.translate(op.upper, scope, timecontext, **kwargs)
        if op.upper is not None
        else float('inf')
    )
    lower = (
        t.translate(op.lower, scope, timecontext, **kwargs)
        if op.lower is not None
        else float('-inf')
    )

    def column_min(value, limit):
        """Given the minimum limit, return values that are greater
        than or equal to this limit."""
        return F.when(value < limit, limit).otherwise(value)

    def column_max(value, limit):
        """Given the maximum limit, return values that are less
        than or equal to this limit."""
        return F.when(value > limit, limit).otherwise(value)

    def clip(column, lower_value, upper_value):
        return column_max(
            column_min(column, F.lit(lower_value)), F.lit(upper_value)
        )

    return clip(col, lower, upper).cast(spark_dtype)


@compiles(ops.Round)
def compile_round(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    scale = (
        t.translate(op.digits, scope, timecontext, raw=True)
        if op.digits is not None
        else 0
    )
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype('long')
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)

    return F.when(src_column == 0, F.lit(0.0)).otherwise(
        F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0)
    )


@compiles(ops.Sqrt)
def compile_sqrt(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    raw_base = t.translate(op.base, scope, timecontext, raw=True)
    try:
        base = float(raw_base)
    except TypeError:
        return F.log(src_column) / F.log(raw_base)
    else:
        return F.log(base, src_column)


@compiles(ops.Ln)
def compile_ln(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    if expr.type() == dtypes.boolean:
        return ~src_column
    return -src_column


@compiles(ops.Add)
def compile_add(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.isnan(src_column) | F.isnull(src_column)


@compiles(ops.IsInf)
def compile_isinf(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return (src_column == float('inf')) | (src_column == float('-inf'))


@compiles(ops.Uppercase)
def compile_uppercase(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.upper(src_column)


@compiles(ops.Lowercase)
def compile_lowercase(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.lower(src_column)


@compiles(ops.Reverse)
def compile_reverse(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.reverse(src_column)


@compiles(ops.Strip)
def compile_strip(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.trim(src_column)


@compiles(ops.LStrip)
def compile_lstrip(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.ltrim(src_column)


@compiles(ops.RStrip)
def compile_rstrip(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.rtrim(src_column)


@compiles(ops.Capitalize)
def compile_capitalize(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.initcap(src_column)


@compiles(ops.Substring)
def compile_substring(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    start = t.translate(op.start, scope, timecontext, raw=True) + 1
    length = t.translate(op.length, scope, timecontext, raw=True)

    if isinstance(start, pyspark.sql.Column) or isinstance(
        length, pyspark.sql.Column
    ):
        raise NotImplementedError(
            "Specifiying Start and length with column expressions "
            "are not supported."
        )

    return src_column.substr(start, length)


@compiles(ops.StringLength)
def compile_string_length(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.length(src_column)


@compiles(ops.StrRight)
def compile_str_right(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @F.udf('string')
    def str_right(s, nchars):
        return s[-nchars:]

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    nchars_column = t.translate(op.nchars, scope, timecontext, **kwargs)
    return str_right(src_column, nchars_column)


@compiles(ops.Repeat)
def compile_repeat(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @F.udf('string')
    def repeat(s, times):
        return s * times

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    times_column = t.translate(op.times, scope, timecontext, **kwargs)
    return repeat(src_column, times_column)


@compiles(ops.StringFind)
def compile_string_find(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @F.udf('long')
    def str_find(s, substr, start, end):
        return s.find(substr, start, end)

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    substr_column = t.translate(op.substr, scope, timecontext, **kwargs)
    start_column = (
        t.translate(op.start, scope, timecontext, **kwargs)
        if op.start
        else F.lit(None)
    )
    end_column = (
        t.translate(op.end, scope, timecontext, **kwargs)
        if op.end
        else F.lit(None)
    )
    return str_find(src_column, substr_column, start_column, end_column)


@compiles(ops.Translate)
def compile_translate(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    from_str = op.from_str.op().value
    to_str = op.to_str.op().value
    return F.translate(src_column, from_str, to_str)


@compiles(ops.LPad)
def compile_lpad(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.lpad(src_column, length, pad)


@compiles(ops.RPad)
def compile_rpad(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.rpad(src_column, length, pad)


@compiles(ops.StringJoin)
def compile_string_join(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @F.udf('string')
    def join(sep, arr):
        return sep.join(arr)

    sep_column = t.translate(op.sep, scope, timecontext, **kwargs)
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    return join(sep_column, F.array(arg))


@compiles(ops.RegexSearch)
def compile_regex_search(t, expr, scope, timecontext, **kwargs):
    import re

    op = expr.op()

    @F.udf('boolean')
    def regex_search(s, pattern):
        return True if re.search(pattern, s) else False

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    pattern = t.translate(op.pattern, scope, timecontext, **kwargs)
    return regex_search(src_column, pattern)


@compiles(ops.RegexExtract)
def compile_regex_extract(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    pattern = op.pattern.op().value
    idx = op.index.op().value
    return F.regexp_extract(src_column, pattern, idx)


@compiles(ops.RegexReplace)
def compile_regex_replace(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    pattern = op.pattern.op().value
    replacement = op.replacement.op().value
    return F.regexp_replace(src_column, pattern, replacement)


@compiles(ops.StringReplace)
def compile_string_replace(*args, **kwargs):
    return compile_regex_replace(*args, **kwargs)


@compiles(ops.StringSplit)
def compile_string_split(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    delimiter = op.delimiter.op().value
    return F.split(src_column, delimiter)


@compiles(ops.StringConcat)
def compile_string_concat(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.concat(*src_columns)


@compiles(ops.StringAscii)
def compile_string_ascii(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.ascii(src_column)


@compiles(ops.StringSQLLike)
def compile_string_like(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    pattern = op.pattern.op().value
    return src_column.like(pattern)


@compiles(ops.ValueList)
def compile_value_list(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    # ignore the `raw` argument when compiling a list, otherwise pyspark fails,
    # because it doesn't automatically upcast literals into expressions
    kwargs.pop("raw", None)
    return [
        t.translate(col, scope, timecontext, raw=False, **kwargs)
        for col in op.values
    ]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='inner')


@compiles(ops.LeftJoin)
def compile_left_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='left')


@compiles(ops.RightJoin)
def compile_right_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='right')


@compiles(ops.OuterJoin)
def compile_outer_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='outer')


@compiles(ops.LeftSemiJoin)
def compile_left_semi_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='leftsemi')


@compiles(ops.LeftAntiJoin)
def compile_left_anti_join(t, expr, scope, timecontext, **kwargs):
    return compile_join(t, expr, scope, timecontext, how='leftanti')


def compile_join(t, expr, scope, timecontext, *, how):
    op = expr.op()

    left_df = t.translate(op.left, scope, timecontext)
    right_df = t.translate(op.right, scope, timecontext)

    pred_columns = []
    for pred in op.predicates:
        pred_op = pred.op()
        if not isinstance(pred_op, ops.Equals):
            raise NotImplementedError(
                "Only equality predicate is supported, but got {}".format(
                    type(pred_op)
                )
            )
        pred_columns.append(pred_op.left.get_name())

    return left_df.join(right_df, pred_columns, how)


@compiles(ops.Distinct)
def compile_distinct(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    return t.translate(op.table, scope, timecontext, **kwargs).distinct()


def _canonicalize_interval(t, interval, scope, timecontext, **kwargs):
    """Convert interval to integer timestamp of second

    When pyspark cast timestamp to integer type, it uses the number of seconds
    since epoch. Therefore, we need cast ibis interval correspondingly.
    """
    if isinstance(interval, ir.IntervalScalar):
        value = t.translate(interval, scope, timecontext, **kwargs)
        # value is in nanoseconds and spark uses seconds since epoch
        return int(value / 1e9)
    elif isinstance(interval, int):
        return interval
    raise com.UnsupportedOperationError(
        f'type {type(interval)} is not supported in preceding /following '
        'in window.'
    )


@compiles(ops.Window)
def compile_window_op(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    window = op.window
    operand = op.expr

    group_by = window._group_by
    grouping_keys = [
        key_op.name
        if isinstance(key_op, ops.TableColumn)
        else t.translate(key, scope, timecontext, **kwargs)
        for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by
    # Timestamp needs to be cast to long for window bounds in spark
    ordering_keys = [
        F.col(sort_expr.get_name()).cast('long')
        if isinstance(sort_expr.op().expr, types.TimestampColumn)
        else sort_expr.get_name()
        for sort_expr in order_by
    ]
    context = AggregationContext.WINDOW
    pyspark_window = Window.partitionBy(grouping_keys).orderBy(ordering_keys)

    # If the operand is a shift op (e.g. lead, lag), Spark will set the window
    # bounds. Only set window bounds here if not a shift operation.
    if not isinstance(operand.op(), ops.ShiftBase):
        if window.preceding is None:
            start = Window.unboundedPreceding
        else:
            start = -_canonicalize_interval(
                t, window.preceding, scope, timecontext, **kwargs
            )
        if window.following is None:
            end = Window.unboundedFollowing
        else:
            end = _canonicalize_interval(
                t, window.following, scope, timecontext, **kwargs
            )

        if (
            isinstance(window.preceding, ir.IntervalScalar)
            or isinstance(window.following, ir.IntervalScalar)
            or window.how == "range"
        ):
            pyspark_window = pyspark_window.rangeBetween(start, end)
        else:
            pyspark_window = pyspark_window.rowsBetween(start, end)

    res_op = operand.op()
    if isinstance(res_op, (ops.NotAll, ops.NotAny)):
        # For NotAll and NotAny, negation must be applied after .over(window)
        # Here we rewrite node to be its negation, and negate it back after
        # translation and window operation
        operand = res_op.negate().to_expr()
    result = t.translate(operand, scope, timecontext, context=context).over(
        pyspark_window
    )

    if isinstance(res_op, (ops.NotAll, ops.NotAny)):
        return ~result
    elif isinstance(res_op, (ops.MinRank, ops.DenseRank, ops.RowNumber)):
        # result must be cast to long type for Rank / RowNumber
        return result.astype('long') - 1
    else:
        return result


def _handle_shift_operation(t, expr, scope, timecontext, *, fn, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    default = op.default.op().value if op.default is not None else op.default
    offset = op.offset.op().value if op.offset is not None else op.offset

    if offset:
        return fn(src_column, count=offset, default=default)
    else:
        return fn(src_column, default=default)


@compiles(ops.Lag)
def compile_lag(t, expr, scope, timecontext, **kwargs):
    return _handle_shift_operation(
        t, expr, scope, timecontext, fn=F.lag, **kwargs
    )


@compiles(ops.Lead)
def compile_lead(t, expr, scope, timecontext, **kwargs):
    return _handle_shift_operation(
        t, expr, scope, timecontext, fn=F.lead, **kwargs
    )


@compiles(ops.MinRank)
def compile_rank(t, expr, scope, timecontext, **kwargs):
    return F.rank()


@compiles(ops.DenseRank)
def compile_dense_rank(t, expr, scope, timecontext, **kwargs):
    return F.dense_rank()


@compiles(ops.PercentRank)
def compile_percent_rank(t, expr, scope, timecontext, **kwargs):
    return F.percent_rank()


@compiles(ops.CumeDist)
def compile_cume_dist(t, expr, scope, timecontext, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support cume_dist with Ibis.'
    )


@compiles(ops.NTile)
def compile_ntile(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    buckets = op.buckets.op().value
    return F.ntile(buckets)


@compiles(ops.FirstValue)
def compile_first_value(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.first(src_column)


@compiles(ops.LastValue)
def compile_last_value(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.last(src_column)


@compiles(ops.NthValue)
def compile_nth_value(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    nth = t.translate(op.nth, scope, timecontext, raw=True)
    return F.nth_value(src_column, nth + 1)


@compiles(ops.RowNumber)
def compile_row_number(t, expr, scope, timecontext, **kwargs):
    return F.row_number()


# -------------------------- Temporal Operations ----------------------------

# Ibis value to PySpark value
_time_unit_mapping = {
    'Y': 'year',
    'Q': 'quarter',
    'M': 'month',
    'W': 'week',
    'D': 'day',
    'h': 'hour',
    'm': 'minute',
    's': 'second',
}


@compiles(ops.Date)
def compile_date(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.to_date(src_column).cast('timestamp')


def _extract_component_from_datetime(
    t, expr, scope, timecontext, *, extract_fn, **kwargs
):
    op = expr.op()
    date_col = t.translate(op.arg, scope, timecontext, **kwargs)
    return extract_fn(date_col).cast('integer')


@compiles(ops.ExtractYear)
def compile_extract_year(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.year, **kwargs
    )


@compiles(ops.ExtractMonth)
def compile_extract_month(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.month, **kwargs
    )


@compiles(ops.ExtractDay)
def compile_extract_day(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.dayofmonth, **kwargs
    )


@compiles(ops.ExtractDayOfYear)
def compile_extract_day_of_year(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.dayofyear, **kwargs
    )


@compiles(ops.ExtractQuarter)
def compile_extract_quarter(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.quarter, **kwargs
    )


@compiles(ops.ExtractEpochSeconds)
def compile_extract_epoch_seconds(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.unix_timestamp, **kwargs
    )


@compiles(ops.ExtractWeekOfYear)
def compile_extract_week_of_year(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.weekofyear, **kwargs
    )


@compiles(ops.ExtractHour)
def compile_extract_hour(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.hour, **kwargs
    )


@compiles(ops.ExtractMinute)
def compile_extract_minute(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.minute, **kwargs
    )


@compiles(ops.ExtractSecond)
def compile_extract_second(t, expr, scope, timecontext, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, timecontext, extract_fn=F.second, **kwargs
    )


@compiles(ops.ExtractMillisecond)
def compile_extract_millisecond(t, expr, scope, timecontext, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support extracting milliseconds.'
    )


@compiles(ops.DateTruncate)
def compile_date_truncate(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    try:
        unit = _time_unit_mapping[op.unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'{op.unit!r} unit is not supported in timestamp truncate'
        )

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.date_trunc(unit, src_column)


@compiles(ops.TimestampTruncate)
def compile_timestamp_truncate(t, expr, scope, timecontext, **kwargs):
    return compile_date_truncate(t, expr, scope, timecontext, **kwargs)


@compiles(ops.Strftime)
def compile_strftime(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    format_str = op.format_str.op().value

    @pandas_udf('string', PandasUDFType.SCALAR)
    def strftime(timestamps):
        return timestamps.dt.strftime(format_str)

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return strftime(src_column)


@compiles(ops.TimestampFromUNIX)
def compile_timestamp_from_unix(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    unixtime = t.translate(op.arg, scope, timecontext, **kwargs)
    if not op.unit:
        return F.to_timestamp(F.from_unixtime(unixtime))
    elif op.unit == 's':
        fmt = 'yyyy-MM-dd HH:mm:ss'
        return F.to_timestamp(F.from_unixtime(unixtime, fmt), fmt)
    else:
        raise com.UnsupportedArgumentError(
            'PySpark backend does not support timestamp from unix time with '
            'unit {}. Supported unit is s.'.format(op.unit)
        )


@compiles(ops.TimestampNow)
def compile_timestamp_now(t, expr, scope, timecontext, **_):
    return F.current_timestamp()


@compiles(ops.StringToTimestamp)
def compile_string_to_timestamp(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    fmt = op.format_str.op().value

    if op.timezone is not None and op.timezone.op().value != "UTC":
        raise com.UnsupportedArgumentError(
            'PySpark backend only supports timezone UTC for converting string '
            'to timestamp.'
        )

    return F.to_timestamp(src_column, fmt)


@compiles(ops.DayOfWeekIndex)
def compile_day_of_week_index(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @pandas_udf('short', PandasUDFType.SCALAR)
    def day_of_week(s):
        return s.dt.dayofweek

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return day_of_week(src_column.cast('timestamp'))


@compiles(ops.DayOfWeekName)
def compiles_day_of_week_name(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def day_name(s):
        return s.dt.day_name()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return day_name(src_column.cast('timestamp'))


def _get_interval_col(
    t, interval_ibis_expr, scope, timecontext, allowed_units=None, **kwargs
):
    # if interval expression is a binary op, translate expression into
    # an interval column and return
    if isinstance(interval_ibis_expr.op(), ops.IntervalBinary):
        return t.translate(interval_ibis_expr, scope, timecontext, **kwargs)

    # otherwise, translate expression into a literal op and construct
    # interval column from literal value and dtype
    if isinstance(interval_ibis_expr.op(), ops.Literal):
        op = interval_ibis_expr.op()
    else:
        op = t.translate(interval_ibis_expr, scope, timecontext, **kwargs).op()

    dtype = op.dtype
    if not isinstance(dtype, dtypes.Interval):
        raise com.UnsupportedArgumentError(
            f'{dtype} expression cannot be converted to interval column. '
            'Must be Interval dtype.'
        )
    if allowed_units and dtype.unit not in allowed_units:
        raise com.UnsupportedArgumentError(
            f'Interval unit "{dtype.unit}" is not allowed. Allowed units are: '
            f'{allowed_units}'
        )

    if isinstance(op.value, pd.Timedelta):
        td_nanos = op.value.value
        if td_nanos % 1000 != 0:
            raise com.UnsupportedArgumentError(
                'Interval with nanoseconds is not supported. The '
                'smallest unit supported by Spark is microseconds.'
            )
        td_micros = td_nanos // 1000
        return F.expr(f'INTERVAL {td_micros} MICROSECOND')
    else:
        return F.expr(f'INTERVAL {op.value} {_time_unit_mapping[dtype.unit]}')


def _compile_datetime_binop(
    t, expr, scope, timecontext, *, fn, allowed_units, **kwargs
):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = _get_interval_col(
        t, op.right, scope, timecontext, allowed_units, **kwargs
    )

    return fn(left, right)


@compiles(ops.DateAdd)
def compile_date_add(t, expr, scope, timecontext, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        timecontext,
        fn=(lambda l, r: (l + r).cast('timestamp')),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.DateSub)
def compile_date_sub(t, expr, scope, timecontext, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        timecontext,
        fn=(lambda l, r: (l - r).cast('timestamp')),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.DateDiff)
def compile_date_diff(t, expr, scope, timecontext, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support DateDiff as there is no '
        'timedelta type.'
    )


@compiles(ops.TimestampAdd)
def compile_timestamp_add(t, expr, scope, timecontext, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D', 'h', 'm', 's']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        timecontext,
        fn=(lambda l, r: (l + r).cast('timestamp')),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampSub)
def compile_timestamp_sub(t, expr, scope, timecontext, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D', 'h', 'm', 's']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        timecontext,
        fn=(lambda l, r: (l - r).cast('timestamp')),
        allowed_units=allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampDiff)
def compile_timestamp_diff(t, expr, scope, timecontext, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support TimestampDiff as there is no '
        'timedelta type.'
    )


def _compile_interval_binop(t, expr, scope, timecontext, *, fn, **kwargs):
    op = expr.op()

    left = _get_interval_col(t, op.left, scope, timecontext, **kwargs)
    right = _get_interval_col(t, op.right, scope, timecontext, **kwargs)

    return fn(left, right)


@compiles(ops.IntervalAdd)
def compile_interval_add(t, expr, scope, timecontext, **kwargs):
    return _compile_interval_binop(
        t, expr, scope, timecontext, fn=(lambda l, r: l + r), **kwargs
    )


@compiles(ops.IntervalSubtract)
def compile_interval_subtract(t, expr, scope, timecontext, **kwargs):
    return _compile_interval_binop(
        t, expr, scope, timecontext, fn=(lambda l, r: l - r), **kwargs
    )


@compiles(ops.IntervalFromInteger)
def compile_interval_from_integer(t, expr, scope, timecontext, **kwargs):
    raise com.UnsupportedOperationError(
        'Interval from integer column is unsupported for the PySpark backend.'
    )


# -------------------------- Array Operations ----------------------------


@compiles(ops.ArrayColumn)
def compile_array_column(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    cols = t.translate(op.cols, scope, timecontext, **kwargs)
    return F.array(cols)


@compiles(ops.ArrayLength)
def compile_array_length(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.size(src_column)


@compiles(ops.ArraySlice)
def compile_array_slice(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    start = op.start.op().value if op.start is not None else op.start
    stop = op.stop.op().value if op.stop is not None else op.stop
    spark_type = ibis_array_dtype_to_spark_dtype(op.arg.type())

    @F.udf(spark_type)
    def array_slice(array):
        return array[start:stop]

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return array_slice(src_column)


@compiles(ops.ArrayIndex)
def compile_array_index(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    index = op.index.op().value + 1
    return F.element_at(src_column, index)


@compiles(ops.ArrayConcat)
def compile_array_concat(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)
    return F.concat(left, right)


@compiles(ops.ArrayRepeat)
def compile_array_repeat(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    times = op.times.op().value
    return F.flatten(F.array_repeat(src_column, times))


@compiles(ops.ArrayCollect)
def compile_array_collect(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.collect_list(src_column)


# --------------------------- Null Operations -----------------------------


@compiles(ops.NullLiteral)
def compile_null_literal(t, expr, scope, timecontext, **kwargs):
    return F.lit(None)


@compiles(ops.IfNull)
def compile_if_null(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    ifnull_col = t.translate(op.ifnull_expr, scope, timecontext, **kwargs)
    return F.when(col.isNull() | F.isnan(col), ifnull_col).otherwise(col)


@compiles(ops.NullIf)
def compile_null_if(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    nullif_col = t.translate(op.null_if_expr, scope, timecontext, **kwargs)
    return F.when(col == nullif_col, F.lit(None)).otherwise(col)


@compiles(ops.IsNull)
def compile_is_null(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.isnull(col) | F.isnan(col)


@compiles(ops.NotNull)
def compile_not_null(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope, timecontext, **kwargs)
    return ~F.isnull(col) & ~F.isnan(col)


@compiles(ops.DropNa)
def compile_dropna_table(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    table = t.translate(op.table, scope, timecontext, **kwargs)
    subset = op.subset
    if subset is not None:
        subset = [col.get_name() for col in subset]
    return table.dropna(how=op.how, subset=subset)


@compiles(ops.FillNa)
def compile_fillna_table(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    table = t.translate(op.table, scope, timecontext, **kwargs)
    raw_replacements = op.replacements
    replacements = (
        dict(raw_replacements)
        if isinstance(raw_replacements, frozendict)
        else raw_replacements.op().value
    )
    return table.fillna(replacements)


# ------------------------- User defined function ------------------------


@compiles(ops.ElementWiseVectorizedUDF)
def compile_elementwise_udf(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    spark_output_type = spark_dtype(op.return_type)
    func = op.func
    spark_udf = pandas_udf(func, spark_output_type, PandasUDFType.SCALAR)
    func_args = (
        t.translate(arg, scope, timecontext, **kwargs) for arg in op.func_args
    )
    return spark_udf(*func_args)


@compiles(ops.ReductionVectorizedUDF)
def compile_reduction_udf(t, expr, scope, timecontext, context=None, **kwargs):
    op = expr.op()

    spark_output_type = spark_dtype(op.return_type)
    spark_udf = pandas_udf(
        op.func, spark_output_type, PandasUDFType.GROUPED_AGG
    )
    func_args = (
        t.translate(arg, scope, timecontext, **kwargs) for arg in op.func_args
    )

    col = spark_udf(*func_args)
    if context:
        return col
    else:
        src_table = t.translate(
            op.func_args[0].op().table, scope, timecontext, **kwargs
        )
        return src_table.agg(col)


@compiles(ops.SearchedCase)
def compile_searched_case(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    existing_when = None

    for case, result in zip(op.cases, op.results):
        if existing_when is not None:
            # Spark allowed chained when statement
            when = existing_when.when
        else:
            when = F.when

        existing_when = when(
            t.translate(case, scope, timecontext, **kwargs),
            t.translate(result, scope, timecontext, **kwargs),
        )

    return existing_when.otherwise(
        t.translate(op.default, scope, timecontext, **kwargs)
    )


@compiles(ops.View)
def compile_view(t, expr, scope, timecontext, session, **kwargs):
    op = expr.op()
    name = op.name
    child = op.child
    tables = session.catalog.listTables()
    if any(name == table.name and not table.isTemporary for table in tables):
        raise ValueError(
            f"table or non-temporary view `{name}` already exists"
        )
    result = t.translate(child, scope, timecontext, session=session, **kwargs)
    result.createOrReplaceTempView(name)
    return result


@compiles(ops.SQLStringView)
def compile_sql_view(t, expr, scope, timecontext, session, **kwargs):
    op = expr.op()
    result = session.sql(op.query)
    result.createOrReplaceTempView(op.name)
    return result


@compiles(ops.StringContains)
def compile_string_contains(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    haystack = t.translate(op.haystack, scope, timecontext, **kwargs)
    needle = t.translate(op.needle, scope, timecontext, **kwargs)
    return haystack.contains(needle)


@compiles(ops.Unnest)
def compile_unnest(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    column = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.explode(column)


@compiles(ops.NullIfZero)
def compile_null_if_zero(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.when(arg == 0, F.lit(None)).otherwise(arg)


@compiles(ops.Acos)
@compiles(ops.Asin)
@compiles(ops.Atan)
@compiles(ops.Cos)
@compiles(ops.Sin)
@compiles(ops.Tan)
def compile_trig(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    func_name = op.__class__.__name__.lower()
    func = getattr(F, func_name)
    return func(arg)


@compiles(ops.Cot)
def compile_cot(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.cos(arg) / F.sin(arg)


@compiles(ops.Atan2)
def compile_atan2(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    y, x = (t.translate(arg, scope, timecontext, **kwargs) for arg in op.args)
    return F.atan2(y, x)


@compiles(ops.Degrees)
def compile_degrees(t, expr, scope, timecontext, **kwargs):
    return F.degrees(t.translate(expr.op().arg, scope, timecontext, **kwargs))


@compiles(ops.Radians)
def compile_radians(t, expr, scope, timecontext, **kwargs):
    return F.radians(t.translate(expr.op().arg, scope, timecontext, **kwargs))


@compiles(ops.ZeroIfNull)
def compile_zero_if_null(t, expr, scope, timecontext, **kwargs):
    col = t.translate(expr.op().arg, scope, timecontext, **kwargs)
    return F.when(col.isNull() | F.isnan(col), F.lit(0)).otherwise(col)


@compiles(ops.Where)
def compile_where(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    return F.when(
        t.translate(op.bool_expr, scope, timecontext, **kwargs),
        t.translate(op.true_expr, scope, timecontext, **kwargs),
    ).otherwise(t.translate(op.false_null_expr, scope, timecontext, **kwargs))


@compiles(ops.RandomScalar)
def compile_random(*args, **kwargs):
    return F.rand()


@compiles(ops.InMemoryTable)
@compiles(PandasInMemoryTable)
def compile_in_memory_table(t, expr, scope, timecontext, session, **kwargs):
    op = expr.op()
    fields = [
        pt.StructField(name, ibis_dtype_to_spark_dtype(dtype), dtype.nullable)
        for name, dtype in op.schema.items()
    ]
    schema = pt.StructType(fields)
    return session.createDataFrame(data=op.data.to_frame(), schema=schema)


@compiles(ops.BitwiseAnd)
def compile_bitwise_and(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)

    return left.bitwiseAND(right)


@compiles(ops.BitwiseOr)
def compile_bitwise_or(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)

    return left.bitwiseOR(right)


@compiles(ops.BitwiseXor)
def compile_bitwise_xor(t, expr, scope, timecontext, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope, timecontext, **kwargs)
    right = t.translate(op.right, scope, timecontext, **kwargs)

    return left.bitwiseXOR(right)


@compiles(ops.BitwiseNot)
def compile_bitwise_not(t, expr, scope, timecontext, **kwargs):
    op = expr.op()
    arg = t.translate(op.arg, scope, timecontext, **kwargs)
    return F.bitwise_not(arg)
