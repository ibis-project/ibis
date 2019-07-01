import ibis.sql.compiler as comp
from ibis.impala.compiler import (
    ImpalaContext,
    ImpalaDialect,
    ImpalaExprTranslator,
    ImpalaSelect,
)


class SparkSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return SparkSelect


class SparkUnion(comp.Union):
    pass


class SparkQueryBuilder(comp.QueryBuilder):
    select_builder = SparkSelectBuilder
    union_class = SparkUnion


def build_ast(expr, context):
    builder = SparkQueryBuilder(expr, context=context)
    return builder.get_result()


class SparkContext(ImpalaContext):
    pass


class SparkExprTranslator(ImpalaExprTranslator):
    context_class = SparkContext


class SparkSelect(ImpalaSelect):
    translator = SparkExprTranslator


class SparkDialect(ImpalaDialect):
    translator = SparkExprTranslator
