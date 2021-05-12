from ibis.backends.base.sql.compiler import (
    Dialect,
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    SelectBuilder,
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


class BaseSelectBuilder(SelectBuilder):
    _select_class = BaseSelect


class BaseQueryBuilder(QueryBuilder):
    select_builder = BaseSelectBuilder


def to_sql(expr, context=None):
    if context is None:
        context = BaseDialect.make_context()
    return BaseQueryBuilder(expr, context).get_result().queries[0].compile()
