import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    Dialect,
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import (
    operation_registry,
    quote_identifier,
)


def build_ast(expr, context):
    assert context is not None, 'context is None'
    builder = BaseQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    assert context is not None, 'context is None'
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    if context is None:
        context = BaseDialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()


# ----------------------------------------------------------------------
# Select compilation


class BaseSelectBuilder(SelectBuilder):
    pass


class BaseQueryBuilder(QueryBuilder):
    pass


class BaseContext(QueryContext):
    def _to_sql(self, expr, ctx):
        return to_sql(expr, ctx)


class BaseSelect(Select):
    pass


class BaseTableSetFormatter(TableSetFormatter):
    def _quote_identifier(self, name):
        return quote_identifier(name)


# TODO move the name method to ExprTranslator and use that instead
class BaseExprTranslator(ExprTranslator):
    """Base expression translator."""

    _registry = operation_registry
    context_class = BaseContext

    @staticmethod
    def _name_expr(formatted_expr, quoted_name):
        return '{} AS {}'.format(formatted_expr, quoted_name)

    def name(self, translated, name, force=True):
        """Return expression with its identifier."""
        return self._name_expr(translated, quote_identifier(name, force=force))


class BaseDialect(Dialect):
    translator = BaseExprTranslator


rewrites = BaseExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
