from ibis.backends.base.sql.compiler import (
    Dialect,
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import quote_identifier


class BaseTableSetFormatter(TableSetFormatter):
    def _quote_identifier(self, name):
        return quote_identifier(name)


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


class BaseSelectBuilder(SelectBuilder):
    _select_class = BaseSelect


class BaseQueryBuilder(QueryBuilder):
    select_builder = BaseSelectBuilder


def build_ast(expr, context):
    return BaseQueryBuilder(expr, context=context).get_result()


def to_sql(expr, context=None):
    if context is None:
        context = BaseDialect.make_context()
    return build_ast(expr, context).queries[0].compile()
