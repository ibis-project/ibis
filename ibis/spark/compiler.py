from functools import partial

import regex as re
import toolz

import ibis
import ibis.expr.lineage as lin
import ibis.sql.compiler as comp
from ibis.impala import compiler as impala_compiler
from ibis.bigquery.compiler import (
    BigQueryUnion,
    BigQueryContext,
    BigQueryExprTranslator,
    BigQueryTableSetFormatter,
    to_sql,
    _extract_field,
)


SparkUDFNode = BigQueryUDFNode

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


#dialect = SparkDialect
