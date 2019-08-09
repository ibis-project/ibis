import collections
import enum
import functools

import pyspark.sql.functions as F
from pyspark.sql.window import Window

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as types
from ibis.pyspark.operations import PySparkTable


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

    def translate(self, expr, **kwargs):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr, **kwargs)
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )


compiles = PySparkExprTranslator.compiles


@compiles(PySparkTable)
def compile_datasource(t, expr):
    op = expr.op()
    name, _, client = op.args
    return client._session.table(name)


def compile_table_and_cache(t, expr, cache):
    """Compile a Table expression and cache the result
    """

    assert isinstance(expr, types.TableExpr)
    if expr in cache:
        table = cache[expr]
    else:
        table = t.translate(expr)
        cache[expr] = table
    return table


@compiles(ops.Selection)
def compile_selection(t, expr, **kwargs):
    # Cache compile results for tables
    table_cache = {}

    op = expr.op()

    src_table = compile_table_and_cache(t, op.table, table_cache)
    col_names_in_selection_order = []

    for selection in op.selections:
        if isinstance(selection, types.TableExpr):
            col_names_in_selection_order.extend(selection.columns)
        elif isinstance(selection, types.ColumnExpr):
            column_name = selection.get_name()
            col_names_in_selection_order.append(column_name)
            column = t.translate(selection, table_cache=table_cache)
            src_table = src_table.withColumn(column_name, column)

    return src_table[col_names_in_selection_order]


@compiles(ops.TableColumn)
def compile_column(t, expr, table_cache={}, **kwargs):
    op = expr.op()
    table = compile_table_and_cache(t, op.table, table_cache)
    return table[op.name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.table)


@compiles(ops.Equals)
def compile_equals(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.left) == t.translate(op.right)


@compiles(ops.Greater)
def compile_greater(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.left) > t.translate(op.right)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.left) >= t.translate(op.right)


@compiles(ops.Multiply)
def compile_multiply(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.left) * t.translate(op.right)


@compiles(ops.Subtract)
def compile_subtract(t, expr, **kwargs):
    op = expr.op()
    return t.translate(op.left) - t.translate(op.right)


@compiles(ops.Literal)
def compile_literal(t, expr, raw=False, **kwargs):
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
    else:
        return F.lit(expr.op().value)


@compiles(ops.Aggregation)
def compile_aggregation(t, expr, **kwargs):
    op = expr.op()

    src_table = t.translate(op.table)

    if op.by:
        context = AggregationContext.GROUP
        aggs = [t.translate(m, context=context)
                for m in op.metrics]
        bys = [t.translate(b) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        context = AggregationContext.ENTIRE
        aggs = [t.translate(m, context=context)
                for m in op.metrics]
        return src_table.agg(*aggs)


@compiles(ops.Contains)
def compile_contains(t, expr, **kwargs):
    op = expr.op()
    col = t.translate(op.value)
    return col.isin(t.translate(op.options))


def compile_aggregator(t, expr, fn, context=None, **kwargs):
    op = expr.op()
    src_col = t.translate(op.arg)

    if getattr(op, "where", None) is not None:
        condition = t.translate(op.where)
        src_col = F.when(condition, src_col)

    col = fn(src_col)
    if context:
        return col
    else:
        return t.translate(expr.op().arg.op().table).select(col)


@compiles(ops.GroupConcat)
def compile_group_concat(t, expr, context=None, **kwargs):
    sep = expr.op().sep.op().value

    def fn(col):
        return F.concat_ws(sep, F.collect_list(col))
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Any)
def compile_any(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.NotAny)
def compile_notany(t, expr, context=None, **kwargs):

    def fn(col):
        return ~F.max(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.All)
def compile_all(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.NotAll)
def compile_notall(t, expr, context=None, **kwargs):

    def fn(col):
        return ~F.min(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Count)
def compile_count(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.count, context)


@compiles(ops.Max)
def compile_max(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.Min)
def compile_min(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.Mean)
def compile_mean(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.mean, context)


@compiles(ops.Sum)
def compile_sum(t, expr, context=None, **kwargs):
    return compile_aggregator(t, expr, F.sum, context)


@compiles(ops.StandardDev)
def compile_std(t, expr, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}"
            .format(how)
        )

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Variance)
def compile_variance(t, expr, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}"
            .format(how)
        )

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, context=None, **kwargs):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError(
            "Does not support 'how': {}".format(how)
        )

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.WindowOp)
def compile_window_op(t, expr, **kwargs):
    op = expr.op()

    return (t.translate(op.expr, context=AggregationContext.WINDOW)
            .over(compile_window(op.window)))


@compiles(ops.Greatest)
def compile_greatest(t, expr, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, expr, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.abs(src_column)


@compiles(ops.Round)
def compile_round(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    scale = t.translate(op.digits, raw=True) if op.digits is not None else 0
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype('long')
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)

    return F.when(src_column == 0, F.lit(0.0)) \
        .otherwise(F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0))


@compiles(ops.Sqrt)
def compile_sqrt(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    # Spark log method only takes float
    return F.log(float(t.translate(op.base, raw=True)), src_column)


@compiles(ops.Ln)
def compile_ln(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, expr, **kwargs):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return -src_column


@compiles(ops.Add)
def compile_add(t, expr, **kwargs):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, expr, **kwargs):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, expr, **kwargs):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, expr, **kwargs):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.isnan(src_column)


@compiles(ops.IsInf)
def compile_isinf(t, expr, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg)
    return (src_column == float('inf')) | (src_column == float('-inf'))


@compiles(ops.ValueList)
def compile_value_list(t, expr, **kwargs):
    op = expr.op()
    return [t.translate(col) for col in op.values]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr, **kwargs):
    return compile_join(t, expr, 'inner')


def compile_join(t, expr, how):
    op = expr.op()

    left_df = t.translate(op.left)
    right_df = t.translate(op.right)
    # TODO: Handle multiple predicates
    predicates = t.translate(op.predicates[0])

    return left_df.join(right_df, predicates, how)


# Cannot register with @compiles because window doesn't have an
# op() object
def compile_window(expr):
    spark_window = Window.partitionBy()
    return spark_window
