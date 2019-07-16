from ibis.spark.client import SparkClient
from ibis.pyspark.operations import PysparkTable
from ibis.pyspark.compiler import translate

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

    def execute(self, df, params=None, limit='default', **kwargs):
        return df.toPandas()
