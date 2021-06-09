from ibis.backends.base.sql.compiler import (
    Dialect,
    ExprTranslator,
    Compiler,
    QueryContext,
    Select,
    TableSetFormatter,
)


class BaseTableSetFormatter(TableSetFormatter):
    pass


class BaseContext(QueryContext):
    pass


class BaseExprTranslator(ExprTranslator):
    context_class = BaseContext


class BaseDialect(Dialect):
    translator = BaseExprTranslator


class BaseSelect(Select):
    translator = BaseExprTranslator
    table_set_formatter = BaseTableSetFormatter


class BaseCompiler(Compiler):
    select_class = BaseSelect
