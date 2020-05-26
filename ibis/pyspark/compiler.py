import collections
import enum
import functools
import operator

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import PandasUDFType, pandas_udf

import ibis.common.exceptions as com
import ibis.expr.datatypes as dtypes
import ibis.expr.operations as ops
import ibis.expr.types as types
from ibis import interval
from ibis.pyspark.operations import PySparkTable
from ibis.spark.compiler import SparkContext, SparkDialect
from ibis.spark.datatypes import (
    ibis_array_dtype_to_spark_dtype,
    ibis_dtype_to_spark_dtype,
)


class PySparkContext(SparkContext):
    pass


class AggregationContext(enum.Enum):
    ENTIRE = 0
    WINDOW = 1
    GROUP = 2


class PySparkExprTranslator:
    _registry = {}
    context_class = PySparkContext

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def translate(self, expr, scope, **kwargs):
        """
        Translate Ibis expression into a PySpark object.

        All translated expressions are cached within scope. If an expression is
        found within scope, it's returned. Otherwise, the it's translated and
        cached for future reference.

        :param expr: ibis expression
        :param scope: dictionary mapping from operation to translated result
        :param kwargs: parameters passed as keyword args (e.g. window)
        :return: translated PySpark DataFrame or Column object
        """
        # The operation node type the typed expression wraps
        op = expr.op()

        if op in scope:
            return scope[op]
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            result = formatter(self, expr, scope, **kwargs)
            scope[op] = result
            return result
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )


class PySparkDialect(SparkDialect):
    translator = PySparkExprTranslator


compiles = PySparkExprTranslator.compiles
dialect = PySparkDialect


@compiles(PySparkTable)
def compile_datasource(t, expr, scope):
    op = expr.op()
    name, _, client = op.args
    return client._session.table(name)


@compiles(ops.SQLQueryResult)
def compile_sql_query_result(t, expr, scope, **kwargs):
    op = expr.op()
    query, _, client = op.args
    return client._session.sql(query)


@compiles(ops.Selection)
def compile_selection(t, expr, scope, **kwargs):
    op = expr.op()

    src_table = t.translate(op.table, scope)

    col_in_selection_order = []
    for selection in op.selections:
        if isinstance(selection, types.TableExpr):
            col_in_selection_order.extend(selection.columns)
        elif isinstance(selection, (types.ColumnExpr, types.ScalarExpr)):
            col = t.translate(selection, scope=scope).alias(
                selection.get_name()
            )
            col_in_selection_order.append(col)
        else:
            raise NotImplementedError(
                f"Unrecoginized type in selections: {type(selection)}"
            )

    if col_in_selection_order:
        src_table = src_table[col_in_selection_order]

    for predicate in op.predicates:
        col = t.translate(predicate, scope)
        src_table = src_table[col]

    if op.sort_keys:
        sort_cols = [t.translate(key, scope) for key in op.sort_keys]

        return src_table.sort(*sort_cols)
    else:
        return src_table


@compiles(ops.SortKey)
def compile_sort_key(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.expr, scope)

    if op.ascending:
        return col.asc()
    else:
        return col.desc()


@compiles(ops.TableColumn)
def compile_column(t, expr, scope, **kwargs):
    op = expr.op()
    table = t.translate(op.table, scope)
    return table[op.name]


@compiles(ops.DistinctColumn)
def compile_distinct(t, expr, scope, **kwargs):
    op = expr.op()
    root_table_expr = op.root_tables()[0].to_expr()
    src_table = t.translate(root_table_expr, scope)
    src_column_name = op.arg.get_name()
    return src_table.select(src_column_name).distinct()[src_column_name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.table, scope)


@compiles(ops.Cast)
def compile_cast(t, expr, scope, **kwargs):
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

    src_column = t.translate(op.arg, scope)
    return src_column.cast(cast_type)


@compiles(ops.Limit)
def compile_limit(t, expr, scope, **kwargs):
    op = expr.op()
    if op.offset != 0:
        raise com.UnsupportedArgumentError(
            'PySpark backend does not support non-zero offset is for '
            'limit operation. Got offset {}.'.format(op.offset)
        )
    df = t.translate(op.table, scope)
    return df.limit(op.n)


@compiles(ops.And)
def compile_and(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) & t.translate(op.right, scope)


@compiles(ops.Or)
def compile_or(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) | t.translate(op.right, scope)


@compiles(ops.Equals)
def compile_equals(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) == t.translate(op.right, scope)


@compiles(ops.NotEquals)
def compile_not_equals(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) != t.translate(op.right, scope)


@compiles(ops.Greater)
def compile_greater(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) > t.translate(op.right, scope)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) >= t.translate(op.right, scope)


@compiles(ops.Less)
def compile_less(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) < t.translate(op.right, scope)


@compiles(ops.LessEqual)
def compile_less_equal(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) <= t.translate(op.right, scope)


@compiles(ops.Multiply)
def compile_multiply(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) * t.translate(op.right, scope)


@compiles(ops.Subtract)
def compile_subtract(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) - t.translate(op.right, scope)


@compiles(ops.Literal)
def compile_literal(t, expr, scope, raw=False, **kwargs):
    """ If raw is True, don't wrap the result with F.lit()
    """
    value = expr.op().value

    if raw:
        return value

    if isinstance(value, collections.abc.Set):
        # Don't wrap set with F.lit
        if isinstance(value, frozenset):
            # Spark doens't like frozenset
            return set(value)
        else:
            return value
    elif isinstance(value, list):
        return F.array(*[F.lit(v) for v in value])
    else:
        return F.lit(expr.op().value)


def _compile_agg(t, agg_expr, scope, context):
    agg = t.translate(agg_expr, scope, context=context)
    if agg_expr.has_name():
        return agg.alias(agg_expr.get_name())
    return agg


@compiles(ops.Aggregation)
def compile_aggregation(t, expr, scope, **kwargs):
    op = expr.op()

    src_table = t.translate(op.table, scope)

    if op.by:
        context = AggregationContext.GROUP
        aggs = [_compile_agg(t, m, scope, context) for m in op.metrics]
        bys = [t.translate(b, scope) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        context = AggregationContext.ENTIRE
        aggs = [_compile_agg(t, m, scope, context) for m in op.metrics]
        return src_table.agg(*aggs)


@compiles(ops.Contains)
def compile_contains(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.value, scope)
    return col.isin(t.translate(op.options, scope))


@compiles(ops.NotContains)
def compile_not_contains(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.value, scope)
    return ~(col.isin(t.translate(op.options, scope)))


def compile_aggregator(t, expr, scope, fn, context=None, **kwargs):
    op = expr.op()
    src_col = t.translate(op.arg, scope)

    if getattr(op, 'where', None) is not None:
        condition = t.translate(op.where, scope)
        src_col = F.when(condition, src_col)

    col = fn(src_col)
    if context is None:
        # We are trying to compile a expr such as some_col.max()
        # to a Spark expression.
        # Here we get the root table df of that column and compile
        # the expr to:
        # df.select(max(some_col))
        return t.translate(expr.op().arg.op().table, scope).select(col)
    elif context == AggregationContext.WINDOW:
        window = kwargs['window']
        return col.over(window)
    else:
        return col


@compiles(ops.GroupConcat)
def compile_group_concat(t, expr, scope, context=None, **kwargs):
    sep = expr.op().sep.op().value

    def fn(col):
        return F.concat_ws(sep, F.collect_list(col))

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Any)
def compile_any(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.max, context, **kwargs)


@compiles(ops.NotAny)
def compile_notany(t, expr, scope, *, context=None, window=None, **kwargs):
    # The code here is a little ugly because the translation are different
    # with different context.
    # When translating col.notany() (context is None), we returns the dataframe
    # so we need to negate the aggregator, i.e., df.select(~F.max(col))
    # When traslating col.notany().over(w), we need to negate the result
    # after the window translation, i.e., ~(F.max(col).over(w))

    if context is None:

        def fn(col):
            return ~(F.max(col))

        return compile_aggregator(t, expr, scope, fn, context, **kwargs)
    else:
        return ~compile_any(
            t, expr, scope, context=context, window=window, **kwargs
        )


@compiles(ops.All)
def compile_all(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.min, context, **kwargs)


@compiles(ops.NotAll)
def compile_notall(t, expr, scope, *, context=None, window=None, **kwargs):
    # See comments for opts.NotAny for reasoning for the if/else
    if context is None:

        def fn(col):
            return ~(F.min(col))

        return compile_aggregator(t, expr, scope, fn, context, **kwargs)
    else:
        return ~compile_all(
            t, expr, scope, context=context, window=window, **kwargs
        )


@compiles(ops.Count)
def compile_count(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.count, context, **kwargs)


@compiles(ops.Max)
@compiles(ops.CumulativeMax)
def compile_max(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.max, context, **kwargs)


@compiles(ops.Min)
@compiles(ops.CumulativeMin)
def compile_min(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.min, context, **kwargs)


@compiles(ops.Mean)
@compiles(ops.CumulativeMean)
def compile_mean(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.mean, context, **kwargs)


@compiles(ops.Sum)
@compiles(ops.CumulativeSum)
def compile_sum(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.sum, context, **kwargs)


@compiles(ops.StandardDev)
def compile_std(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}".format(how)
        )

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Variance)
def compile_variance(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}".format(how)
        )

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError("Does not support 'how': {}".format(how))

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Greatest)
def compile_greatest(t, expr, scope, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, expr, scope, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.abs(src_column)


@compiles(ops.Round)
def compile_round(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    scale = (
        t.translate(op.digits, scope, raw=True) if op.digits is not None else 0
    )
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype('long')
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)

    return F.when(src_column == 0, F.lit(0.0)).otherwise(
        F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0)
    )


@compiles(ops.Sqrt)
def compile_sqrt(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    # Spark log method only takes float
    return F.log(float(t.translate(op.base, scope, raw=True)), src_column)


@compiles(ops.Ln)
def compile_ln(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return -src_column


@compiles(ops.Add)
def compile_add(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.isnan(src_column)


@compiles(ops.IsInf)
def compile_isinf(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return (src_column == float('inf')) | (src_column == float('-inf'))


@compiles(ops.Uppercase)
def compile_uppercase(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.upper(src_column)


@compiles(ops.Lowercase)
def compile_lowercase(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.lower(src_column)


@compiles(ops.Reverse)
def compile_reverse(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.reverse(src_column)


@compiles(ops.Strip)
def compile_strip(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.trim(src_column)


@compiles(ops.LStrip)
def compile_lstrip(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.ltrim(src_column)


@compiles(ops.RStrip)
def compile_rstrip(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.rtrim(src_column)


@compiles(ops.Capitalize)
def compile_capitalize(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.initcap(src_column)


@compiles(ops.Substring)
def compile_substring(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def substring(s, start, length):
        end = start + length
        return s[start:end]

    src_column = t.translate(op.arg, scope)
    start_column = t.translate(op.start, scope)
    length_column = t.translate(op.length, scope)
    return substring(src_column, start_column, length_column)


@compiles(ops.StringLength)
def compile_string_length(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.length(src_column)


@compiles(ops.StrRight)
def compile_str_right(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def str_right(s, nchars):
        return s[-nchars:]

    src_column = t.translate(op.arg, scope)
    nchars_column = t.translate(op.nchars, scope)
    return str_right(src_column, nchars_column)


@compiles(ops.Repeat)
def compile_repeat(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def repeat(s, times):
        return s * times

    src_column = t.translate(op.arg, scope)
    times_column = t.translate(op.times, scope)
    return repeat(src_column, times_column)


@compiles(ops.StringFind)
def compile_string_find(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('long')
    def str_find(s, substr, start, end):
        return s.find(substr, start, end)

    src_column = t.translate(op.arg, scope)
    substr_column = t.translate(op.substr, scope)
    start_column = t.translate(op.start, scope) if op.start else F.lit(None)
    end_column = t.translate(op.end, scope) if op.end else F.lit(None)
    return str_find(src_column, substr_column, start_column, end_column)


@compiles(ops.Translate)
def compile_translate(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    from_str = op.from_str.op().value
    to_str = op.to_str.op().value
    return F.translate(src_column, from_str, to_str)


@compiles(ops.LPad)
def compile_lpad(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.lpad(src_column, length, pad)


@compiles(ops.RPad)
def compile_rpad(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.rpad(src_column, length, pad)


@compiles(ops.StringJoin)
def compile_string_join(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def join(sep, arr):
        return sep.join(arr)

    sep_column = t.translate(op.sep, scope)
    arg = t.translate(op.arg, scope)
    return join(sep_column, F.array(arg))


@compiles(ops.RegexSearch)
def compile_regex_search(t, expr, scope, **kwargs):
    import re

    op = expr.op()

    @F.udf('boolean')
    def regex_search(s, pattern):
        return True if re.search(pattern, s) else False

    src_column = t.translate(op.arg, scope)
    pattern = t.translate(op.pattern, scope)
    return regex_search(src_column, pattern)


@compiles(ops.RegexExtract)
def compile_regex_extract(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    idx = op.index.op().value
    return F.regexp_extract(src_column, pattern, idx)


@compiles(ops.RegexReplace)
def compile_regex_replace(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    replacement = op.replacement.op().value
    return F.regexp_replace(src_column, pattern, replacement)


@compiles(ops.StringReplace)
def compile_string_replace(t, expr, scope, **kwargs):
    return compile_regex_replace(t, expr)


@compiles(ops.StringSplit)
def compile_string_split(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    delimiter = op.delimiter.op().value
    return F.split(src_column, delimiter)


@compiles(ops.StringConcat)
def compile_string_concat(t, expr, scope, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope)
    return F.concat(*src_columns)


@compiles(ops.StringAscii)
def compile_string_ascii(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.ascii(src_column)


@compiles(ops.StringSQLLike)
def compile_string_like(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    return src_column.like(pattern)


@compiles(ops.ValueList)
def compile_value_list(t, expr, scope, **kwargs):
    op = expr.op()
    return [t.translate(col, scope) for col in op.values]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr, scope, **kwargs):
    return compile_join(t, expr, scope, 'inner')


@compiles(ops.LeftJoin)
def compile_left_join(t, expr, scope, **kwargs):
    return compile_join(t, expr, scope, 'left')


@compiles(ops.RightJoin)
def compile_right_join(t, expr, scope, **kwargs):
    return compile_join(t, expr, scope, 'right')


@compiles(ops.OuterJoin)
def compile_outer_join(t, expr, scope, **kwargs):
    return compile_join(t, expr, scope, 'outer')


def compile_join(t, expr, scope, how):
    op = expr.op()

    left_df = t.translate(op.left, scope)
    right_df = t.translate(op.right, scope)

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


@compiles(ops.WindowOp)
def compile_window_op(t, expr, scope, **kwargs):
    op = expr.op()
    window = op.window
    operand = op.expr

    group_by = window._group_by
    grouping_keys = [
        key_op.name
        if isinstance(key_op, ops.TableColumn)
        else t.translate(key, scope)
        for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by
    ordering_keys = [
        key.to_expr().get_name()
        for key in map(operator.methodcaller('op'), order_by)
    ]

    context = AggregationContext.WINDOW
    pyspark_window = Window.partitionBy(grouping_keys).orderBy(ordering_keys)

    # If the operand is a shift op (e.g. lead, lag), Spark will set the window
    # bounds. Only set window bounds here if not a shift operation.
    if not isinstance(operand.op(), ops.ShiftBase):
        start = (
            -window.preceding
            if window.preceding is not None
            else Window.unboundedPreceding
        )
        end = (
            window.following
            if window.following is not None
            else Window.unboundedFollowing
        )
        pyspark_window = pyspark_window.rowsBetween(start, end)

    result = t.translate(
        operand, scope, window=pyspark_window, context=context
    )

    return result


def _handle_shift_operation(t, expr, scope, fn, *, window, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    default = op.default.op().value if op.default is not None else op.default
    offset = op.offset.op().value if op.offset is not None else op.offset

    if offset:
        return fn(src_column, count=offset, default=default).over(window)
    else:
        return fn(src_column, default=default).over(window)


@compiles(ops.Lag)
def compile_lag(t, expr, scope, *, window, **kwargs):
    return _handle_shift_operation(
        t, expr, scope, F.lag, window=window, **kwargs
    )


@compiles(ops.Lead)
def compile_lead(t, expr, scope, *, window, **kwargs):
    return _handle_shift_operation(
        t, expr, scope, F.lead, window=window, **kwargs
    )


@compiles(ops.MinRank)
def compile_rank(t, expr, scope, *, window, **kwargs):
    return F.rank().over(window).astype('long') - 1


@compiles(ops.DenseRank)
def compile_dense_rank(t, expr, scope, *, window, **kwargs):
    return F.dense_rank().over(window).astype('long') - 1


@compiles(ops.PercentRank)
def compile_percent_rank(t, expr, scope, *, window, **kwargs):
    raise com.UnsupportedOperationError(
        'Pyspark percent_rank() function indexes from 0 '
        'instead of 1, and does not match expected '
        'output of ibis expressions.'
    )


@compiles(ops.NTile)
def compile_ntile(t, expr, scope, *, window, **kwargs):
    op = expr.op()
    buckets = op.buckets.op().value
    return F.ntile(buckets).over(window)


@compiles(ops.FirstValue)
def compile_first_value(t, expr, scope, *, window, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope)
    return F.first(src_column).over(window)


@compiles(ops.LastValue)
def compile_last_value(t, expr, scope, *, window, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope)
    return F.last(src_column).over(window)


@compiles(ops.RowNumber)
def compile_row_number(t, expr, scope, *, window, **kwargs):
    return F.row_number().over(window).cast('long') - 1


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
def compile_date(t, expr, scope, **kwargs):
    op = expr.op()
    src_column = t.translate(op.arg, scope)
    return F.to_date(src_column).cast('timestamp')


def _extract_component_from_datetime(t, expr, scope, extract_fn, **kwargs):
    op = expr.op()
    date_col = t.translate(op.arg, scope)
    return extract_fn(date_col)


@compiles(ops.ExtractYear)
def compile_extract_year(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(t, expr, scope, F.year, **kwargs)


@compiles(ops.ExtractMonth)
def compile_extract_month(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(t, expr, scope, F.month, **kwargs)


@compiles(ops.ExtractDay)
def compile_extract_day(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, F.dayofmonth, **kwargs
    )


@compiles(ops.ExtractDayOfYear)
def compile_extract_day_of_year(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(
        t, expr, scope, F.dayofyear, **kwargs
    )


@compiles(ops.ExtractHour)
def compile_extract_hour(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(t, expr, scope, F.hour, **kwargs)


@compiles(ops.ExtractMinute)
def compile_extract_minute(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(t, expr, scope, F.minute, **kwargs)


@compiles(ops.ExtractSecond)
def compile_extract_second(t, expr, scope, **kwargs):
    return _extract_component_from_datetime(t, expr, scope, F.second, **kwargs)


@compiles(ops.ExtractMillisecond)
def compile_extract_millisecond(t, expr, scope, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support extracting milliseconds.'
    )


@compiles(ops.DateTruncate)
def compile_date_truncate(t, expr, scope, **kwargs):
    op = expr.op()

    try:
        unit = _time_unit_mapping[op.unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            '{!r} unit is not supported in timestamp truncate'.format(op.unit)
        )

    src_column = t.translate(op.arg, scope)
    return F.date_trunc(unit, src_column)


@compiles(ops.TimestampTruncate)
def compile_timestamp_truncate(t, expr, scope, **kwargs):
    return compile_date_truncate(t, expr, scope, **kwargs)


@compiles(ops.Strftime)
def compile_strftime(t, expr, scope, **kwargs):
    op = expr.op()
    format_str = op.format_str.op().value

    @pandas_udf('string', PandasUDFType.SCALAR)
    def strftime(timestamps):
        return timestamps.dt.strftime(format_str)

    src_column = t.translate(op.arg, scope)
    return strftime(src_column)


@compiles(ops.TimestampFromUNIX)
def compile_timestamp_from_unix(t, expr, scope, **kwargs):
    op = expr.op()
    unixtime = t.translate(op.arg, scope)
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
def compile_timestamp_now(t, expr, scope, **kwargs):
    return F.current_timestamp()


@compiles(ops.StringToTimestamp)
def compile_string_to_timestamp(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    fmt = op.format_str.op().value

    if op.timezone is not None and op.timezone.op().value != "UTC":
        raise com.UnsupportedArgumentError(
            'PySpark backend only supports timezone UTC for converting string '
            'to timestamp.'
        )

    return F.to_timestamp(src_column, fmt)


@compiles(ops.DayOfWeekIndex)
def compile_day_of_week_index(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('short', PandasUDFType.SCALAR)
    def day_of_week(s):
        return s.dt.dayofweek

    src_column = t.translate(op.arg, scope)
    return day_of_week(src_column.cast('timestamp'))


@compiles(ops.DayOfWeekName)
def compiles_day_of_week_name(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def day_name(s):
        return s.dt.day_name()

    src_column = t.translate(op.arg, scope)
    return day_name(src_column.cast('timestamp'))


def _get_interval_col(t, interval_ibis_expr, scope, allowed_units=None):
    # if interval expression is a binary op, translate expression into
    # an interval column and return
    if isinstance(interval_ibis_expr.op(), ops.IntervalBinaryOp):
        return t.translate(interval_ibis_expr, scope)

    # otherwise, translate expression into a literal op and construct
    # interval column from literal value and dtype
    if isinstance(interval_ibis_expr.op(), ops.Literal):
        op = interval_ibis_expr.op()
    else:
        op = t.translate(interval_ibis_expr, scope).op()

    dtype = op.dtype
    if not isinstance(dtype, dtypes.Interval):
        raise com.UnsupportedArgumentError(
            '{} expression cannot be converted to interval column. '
            'Must be Interval dtype.'.format(dtype)
        )
    if allowed_units and dtype.unit not in allowed_units:
        raise com.UnsupportedArgumentError(
            'Interval unit "{}" is not allowed. Allowed units are: '
            '{}'.format(dtype.unit, allowed_units)
        )
    return F.expr(
        'INTERVAL {} {}'.format(op.value, _time_unit_mapping[dtype.unit])
    )


def _compile_datetime_binop(t, expr, scope, fn, allowed_units, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = _get_interval_col(t, op.right, scope, allowed_units)

    return fn(left, right)


@compiles(ops.DateAdd)
def compile_date_add(t, expr, scope, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        (lambda l, r: (l + r).cast('timestamp')),
        allowed_units,
        **kwargs,
    )


@compiles(ops.DateSub)
def compile_date_sub(t, expr, scope, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        (lambda l, r: (l - r).cast('timestamp')),
        allowed_units,
        **kwargs,
    )


@compiles(ops.DateDiff)
def compile_date_diff(t, expr, scope, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support DateDiff as there is no '
        'timedelta type.'
    )


@compiles(ops.TimestampAdd)
def compile_timestamp_add(t, expr, scope, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D', 'h', 'm', 's']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        (lambda l, r: (l + r).cast('timestamp')),
        allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampSub)
def compile_timestamp_sub(t, expr, scope, **kwargs):
    allowed_units = ['Y', 'W', 'M', 'D', 'h', 'm', 's']
    return _compile_datetime_binop(
        t,
        expr,
        scope,
        (lambda l, r: (l - r).cast('timestamp')),
        allowed_units,
        **kwargs,
    )


@compiles(ops.TimestampDiff)
def compile_timestamp_diff(t, expr, scope, **kwargs):
    raise com.UnsupportedOperationError(
        'PySpark backend does not support TimestampDiff as there is no '
        'timedelta type.'
    )


def _compile_interval_binop(t, expr, scope, fn, **kwargs):
    op = expr.op()

    left = _get_interval_col(t, op.left, scope)
    right = _get_interval_col(t, op.right, scope)

    return fn(left, right)


@compiles(ops.IntervalAdd)
def compile_interval_add(t, expr, scope, **kwargs):
    return _compile_interval_binop(
        t, expr, scope, (lambda l, r: l + r), **kwargs
    )


@compiles(ops.IntervalSubtract)
def compile_interval_subtract(t, expr, scope, **kwargs):
    return _compile_interval_binop(
        t, expr, scope, (lambda l, r: l - r), **kwargs
    )


@compiles(ops.IntervalFromInteger)
def compile_interval_from_integer(t, expr, scope, **kwargs):
    raise com.UnsupportedOperationError(
        'Interval from integer column is unsupported for the PySpark backend.'
    )


# -------------------------- Array Operations ----------------------------


@compiles(ops.ArrayLength)
def compile_array_length(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.size(src_column)


@compiles(ops.ArraySlice)
def compile_array_slice(t, expr, scope, **kwargs):
    op = expr.op()
    start = op.start.op().value if op.start is not None else op.start
    stop = op.stop.op().value if op.stop is not None else op.stop
    spark_type = ibis_array_dtype_to_spark_dtype(op.arg.type())

    @F.udf(spark_type)
    def array_slice(array):
        return array[start:stop]

    src_column = t.translate(op.arg, scope)
    return array_slice(src_column)


@compiles(ops.ArrayIndex)
def compile_array_index(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    index = op.index.op().value + 1
    return F.element_at(src_column, index)


@compiles(ops.ArrayConcat)
def compile_array_concat(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return F.concat(left, right)


@compiles(ops.ArrayRepeat)
def compile_array_repeat(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    times = op.times.op().value
    return F.flatten(F.array_repeat(src_column, times))


@compiles(ops.ArrayCollect)
def compile_array_collect(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.collect_list(src_column)


# --------------------------- Null Operations -----------------------------


@compiles(ops.NullLiteral)
def compile_null_literal(t, expr, scope):
    return F.lit(None)


@compiles(ops.IfNull)
def compile_if_null(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope)
    ifnull_col = t.translate(op.ifnull_expr, scope)
    return F.when(col.isNull(), ifnull_col).otherwise(col)


@compiles(ops.NullIf)
def compile_null_if(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope)
    nullif_col = t.translate(op.null_if_expr, scope)
    return F.when(col == nullif_col, F.lit(None)).otherwise(col)


@compiles(ops.IsNull)
def compile_is_null(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope)
    return F.isnull(col) | F.isnan(col)


@compiles(ops.NotNull)
def compile_not_null(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.arg, scope)
    return ~F.isnull(col) & ~F.isnan(col)


# ------------------------- User defined function ------------------------


@compiles(ops.ElementWiseVectorizedUDF)
def compile_elementwise_udf(t, expr, scope):
    op = expr.op()
    spark_output_type = ibis_dtype_to_spark_dtype(op._output_type)
    spark_udf = pandas_udf(op.func, spark_output_type, PandasUDFType.SCALAR)
    func_args = (t.translate(arg, scope) for arg in op.func_args)
    return spark_udf(*func_args)
