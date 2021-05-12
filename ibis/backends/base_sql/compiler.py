from ibis.backends.base.sql.compiler import (
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
    context = context or BaseContext()
    return build_ast(expr, context).queries[0].compile()
