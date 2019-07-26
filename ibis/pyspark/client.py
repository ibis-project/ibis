import ibis.expr.types as types
from ibis.pyspark.compiler import translate
from ibis.pyspark.operations import PysparkTable
from ibis.spark.client import SparkClient


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
        elif isinstance(expr, types.ScalarExpr):
            return self.compile(expr).toPandas().iloc[0, 0]
        else:
            raise ValueError("Unexpected type: ", type(expr))
