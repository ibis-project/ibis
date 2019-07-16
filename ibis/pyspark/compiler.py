import ibis.common as com
import ibis.sql.compiler as comp
import ibis.expr.operations as ops


from ibis.pyspark.operations import PysparkTable
from ibis.sql.compiler import Dialect

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
    src_table = t.translate(op.selections[0])
    for selection in op.selections[1:]:
        column_name = selection.get_name()
        column = t.translate(selection)
        src_table = src_table.withColumn(column_name, column)

    return src_table


@compiles(ops.TableColumn)
def compile_column(t, expr):
    op = expr.op()
    return t.translate(op.table)[op.name]


t = PysparkExprTranslator()

def translate(expr):
    return t.translate(expr)
