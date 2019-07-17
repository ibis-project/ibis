import ibis.common as com
import ibis.sql.compiler as comp
import ibis.expr.window as window
import ibis.expr.operations as ops
import ibis.expr.types as types


from ibis.pyspark.operations import PysparkTable

from ibis.sql.compiler import Dialect

import pyspark.sql.functions as F
from pyspark.sql.window import Window

_operation_registry = {
}

class PysparkExprTranslator:
    _registry = _operation_registry

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def translate(self, expr):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr)
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


@compiles(ops.Multiply)
def compile_multiply(t, expr):
    op = expr.op()
    return t.translate(op.left) * t.translate(op.right)


@compiles(ops.Subtract)
def compile_subtract(t, expr):
    op = expr.op()
    return t.translate(op.left) - t.translate(op.right)


@compiles(ops.Aggregation)
def compile_aggregation(t, expr):
    op = expr.op()

    src_table = t.translate(op.table)
    aggs = [t.translate(m) for m in op.metrics]

    if op.by:
        bys = [t.translate(b) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        return src_table.agg(*aggs)


@compiles(ops.Max)
def compile_max(t, expr):
    op = expr.op()

    # TODO: Derive the UDF output type from schema
    @F.pandas_udf('long', F.PandasUDFType.GROUPED_AGG)
    def max(v):
        return v.max()

    src_column = t.translate(op.arg)
    return max(src_column)


@compiles(ops.Mean)
def compile_mean(t, expr):
    op = expr.op()
    src_column = t.translate(op.arg)

    return F.mean(src_column)


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


# Cannot register with @compiles because window doesn't have an
# op() object
def compile_window(expr):
    window = expr
    spark_window = Window.partitionBy()
    return spark_window


t = PysparkExprTranslator()

def translate(expr):
    return t.translate(expr)
