from pyspark.sql.column import Column

import ibis.common.exceptions as com
import ibis.expr.types as types
from ibis.pyspark.compiler import PySparkDialect, PySparkExprTranslator
from ibis.pyspark.operations import PySparkTable
from ibis.spark.client import SparkClient


class PySparkClient(SparkClient):
    """
    An ibis client that uses PySpark SQL Dataframe
    """

    dialect = PySparkDialect
    table_class = PySparkTable

    def __init__(self, session):
        super().__init__(session)
        self.translator = PySparkExprTranslator()

    def compile(self, expr, params=None, *args, **kwargs):
        """Compile an ibis expression to a PySpark DataFrame object
        """

        # Insert params in scope
        if params is None:
            scope = {}
        else:
            scope = dict(
                (param.op(), raw_value) for param, raw_value in params.items()
            )
        return self.translator.translate(expr, scope=scope)

    def execute(self, expr, params=None, limit='default', **kwargs):
        if isinstance(expr, types.TableExpr):
            return self.compile(expr, params, **kwargs).toPandas()
        elif isinstance(expr, types.ColumnExpr):
            # expression must be named for the projection
            expr = expr.name('tmp')
            return self.compile(
                expr.to_projection(), params, **kwargs
            ).toPandas()['tmp']
        elif isinstance(expr, types.ScalarExpr):
            compiled = self.compile(expr, params, **kwargs)
            if isinstance(compiled, Column):
                # attach result column to a fake DataFrame and
                # select the result
                compiled = self._session.range(0, 1).select(compiled)
            return compiled.toPandas().iloc[0, 0]
        else:
            raise com.IbisError(
                "Cannot execute expression of type: {}".format(type(expr))
            )
