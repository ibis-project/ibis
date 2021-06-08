from ibis.backends.base.sql.compiler import (
    Dialect,
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    TableSetFormatter,
)


class BaseTableSetFormatter(TableSetFormatter):
    pass


class BaseContext(QueryContext):
    def _to_sql(self, expr, ctx):
        return to_sql(expr, ctx)


class BaseExprTranslator(ExprTranslator):
    context_class = BaseContext


class BaseDialect(Dialect):
    translator = BaseExprTranslator


class BaseSelect(Select):
    translator = BaseExprTranslator
    table_set_formatter = BaseTableSetFormatter


class BaseQueryBuilder(QueryBuilder):
    select_class = BaseSelect


build_ast = BaseQueryBuilder.to_ast


def to_sql(expr, context=None):
    if context is None:
        context = BaseDialect.make_context()
    return BaseQueryBuilder.to_sql(expr, context)
