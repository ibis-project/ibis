import ibis.sql.compiler as comp
from ibis.bigquery.compiler import (
    BigQueryContext,
    BigQueryExprTranslator,
    BigQueryTableSetFormatter,
    BigQueryUnion,
)
from ibis.impala import compiler as impala_compiler


class SparkSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return SparkSelect


SparkUnion = BigQueryUnion


class SparkQueryBuilder(comp.QueryBuilder):
    select_builder = SparkSelectBuilder
    union_class = SparkUnion


def build_ast(expr, context):
    builder = SparkQueryBuilder(expr, context=context)
    return builder.get_result()


SparkContext = BigQueryContext


class SparkExprTranslator(BigQueryExprTranslator):
    context_class = SparkContext


SparkTableSetFormatter = BigQueryTableSetFormatter


class SparkSelect(impala_compiler.ImpalaSelect):
    translator = SparkExprTranslator

    @property
    def table_set_formatter(self):
        return SparkTableSetFormatter


class SparkDialect(impala_compiler.ImpalaDialect):
    translator = SparkExprTranslator
