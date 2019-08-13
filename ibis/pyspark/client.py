from pyspark.sql.column import Column

import ibis.common.exceptions as com
import ibis.expr.types as types
from ibis.pyspark.compiler import PySparkExprTranslator
from ibis.pyspark.operations import PySparkTable
from ibis.spark.client import SparkClient


class PySparkClient(SparkClient):
    """
    An ibis client that uses PySpark SQL Dataframe
    """

    table_class = PySparkTable

    def __init__(self, session):
        super().__init__(session)
        self.translator = PySparkExprTranslator()

    def compile(self, expr, *args, **kwargs):
        """Compile an ibis expression to a PySpark DataFrame object
        """
        return self.translator.translate(expr, scope={})

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
                compiled = self._session.range(0, 1).select(compiled)
            return compiled.toPandas().iloc[0, 0]
        else:
            raise com.IbisError(
                "Cannot execute expression of type: {}".format(type(expr)))

    def sql(self, query):
        raise NotImplementedError(
            "PySpark backend doesn't support sql query")
