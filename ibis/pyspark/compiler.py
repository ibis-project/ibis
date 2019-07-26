import collections
import functools

import pyspark.sql.functions as F
from pyspark.sql.window import Window

import ibis.common as com
import ibis.expr.operations as ops
import ibis.expr.types as types
from ibis.pyspark.operations import PysparkTable
from ibis.sql.compiler import Dialect

_operation_registry = {}


class PysparkExprTranslator:
    _registry = _operation_registry

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


class PysparkDialect(Dialect):
    translator = PysparkExprTranslator


compiles = PysparkExprTranslator.compiles


@compiles(PysparkTable)
def compile_datasource(t, expr):
    op = expr.op()
    name, _, client = op.args
    return client._session.table(name)


@compiles(ops.Selection)
def compile_selection(t, expr):
    op = expr.op()

    if isinstance(op.selections[0], types.ColumnExpr):
        column_names = [expr.op().name for expr in op.selections]
        src_table = t.translate(op.table)[column_names]
    elif isinstance(op.selections[0], types.TableExpr):
        src_table = t.translate(op.table)
        for selection in op.selections[1:]:
            column_name = selection.get_name()
            column = t.translate(selection)
            src_table = src_table.withColumn(column_name, column)

    return src_table


@compiles(ops.TableColumn)
def compile_column(t, expr):
    op = expr.op()
    return t.translate(op.table)[op.name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr):
    op = expr.op()
    return t.translate(op.table)


@compiles(ops.Equals)
def compile_equals(t, expr):
    op = expr.op()
    return t.translate(op.left) == t.translate(op.right)


@compiles(ops.Greater)
def compile_greater(t, expr):
    op = expr.op()
    return t.translate(op.left) > t.translate(op.right)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr):
    op = expr.op()
    return t.translate(op.left) >= t.translate(op.right)


@compiles(ops.Multiply)
def compile_multiply(t, expr):
    op = expr.op()
    return t.translate(op.left) * t.translate(op.right)


@compiles(ops.Subtract)
def compile_subtract(t, expr):
    op = expr.op()
    return t.translate(op.left) - t.translate(op.right)


@compiles(ops.Literal)
def compile_literal(t, expr):
    value = expr.op().value

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
def compile_aggregation(t, expr):
    op = expr.op()

    src_table = t.translate(op.table)
    aggs = [t.translate(m, context="agg")
            for m in op.metrics]

    if op.by:
        bys = [t.translate(b) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        return src_table.agg(*aggs)


@compiles(ops.Contains)
def compile_contains(t, expr):
    col = t.translate(expr.op().value)
    return col.isin(t.translate(expr.op().options))


def compile_aggregator(t, expr, fn, context=None):
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
def compile_group_concat(t, expr, context=None):
    sep = expr.op().sep.op().value

    def fn(col):
        return F.concat_ws(sep, F.collect_list(col))
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Any)
def compile_any(t, expr, context=None):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.NotAny)
def compile_notany(t, expr, context=None):

    def fn(col):
        return ~F.max(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.All)
def compile_all(t, expr, context=None):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.NotAll)
def compile_notall(t, expr, context=None):

    def fn(col):
        return ~F.min(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Count)
def compile_count(t, expr, context=None):
    return compile_aggregator(t, expr, F.count, context)


@compiles(ops.Max)
def compile_max(t, expr, context=None):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.Min)
def compile_min(t, expr, context=None):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.Mean)
def compile_mean(t, expr, context=None):
    return compile_aggregator(t, expr, F.mean, context)


@compiles(ops.Sum)
def compile_sum(t, expr, context=None):
    return compile_aggregator(t, expr, F.sum, context)


@compiles(ops.StandardDev)
def compile_std(t, expr, context=None):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise AssertionError("Unexpected how: {}".format(how))

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Variance)
def compile_variance(t, expr, context=None):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise AssertionError("Unexpected how: {}".format(how))

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, context=None):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.WindowOp)
def compile_window_op(t, expr):
    op = expr.op()
    return t.translate(op.expr).over(compile_window(op.window))


@compiles(ops.Greatest)
def compile_greatest(t, expr):
    op = expr.op()

    src_columns = t.translate(op.arg)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.ValueList)
def compile_value_list(t, expr):
    op = expr.op()
    return [t.translate(col) for col in op.values]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr):
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


t = PysparkExprTranslator()


def translate(expr):
    return t.translate(expr)
