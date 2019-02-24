from ibis.impala import compiler as impala_compiler
from ibis.impala.compiler import ImpalaSelect
import ibis.sql.compiler as comp


class SparkSQLExprTranslator(impala_compiler.ImpalaExprTranslator):
    pass


class SparkSQLSelect(ImpalaSelect):

    translator = SparkSQLExprTranslator


class SparkSQLSelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return SparkSQLSelect


class SparkSQLQueryBuilder(comp.QueryBuilder):

    select_builder = SparkSQLSelectBuilder
