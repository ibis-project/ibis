import ibis.expr.types as types
from ibis.pyspark.compiler import translate
from ibis.pyspark.operations import PysparkTable
from ibis.spark.client import SparkClient

from pyspark.sql.column import Column


class PysparkClient(SparkClient):
    """
    An ibis client that uses Pyspark SQL Dataframe
    """

    dialect = None
    table_class = PysparkTable

    def compile(self, expr, *args, **kwargs):
        """Compile an ibis expression to a Pyspark DataFrame object
        """
        return translate(expr)

    def execute(self, expr, params=None, limit='default', **kwargs):

        if isinstance(expr, types.TableExpr):
            return self.compile(expr).toPandas()
        elif isinstance(expr, types.ColumnExpr):
            # expression must be named for the projection
            expr = expr.name('tmp')
            return self.compile(expr.to_projection()).toPandas()['tmp']
        elif isinstance(expr, types.ScalarExpr):
            compiled = self.compile(expr)
            if isinstance(compiled, Column):
                # attach result column to a fake DataFrame and
                # select the result
                compiled = self._session.range(0, 1) \
                    .select(compiled)
            return compiled.toPandas().iloc[0, 0]
        else:
            raise ValueError("Unexpected type: ", type(expr))
