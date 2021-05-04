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


class BaseTableSetFormatter(TableSetFormatter):
    def _quote_identifier(self, name):
        return quote_identifier(name)


class BaseContext(QueryContext):
    def _to_sql(self, expr, ctx):
        return to_sql(expr, ctx)


class BaseExprTranslator(ExprTranslator):
    _registry = operation_registry
    context_class = BaseContext

    def name(self, translated, name, force=True):
        return '{} AS {}'.format(
            translated, quote_identifier(name, force=force)
        )


@BaseExprTranslator.rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


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
