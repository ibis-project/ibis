from pyspark.sql.column import Column

import ibis.common.exceptions as com
import ibis.expr.types as types
from ibis.backends.spark.client import SparkClient
from ibis.expr.scope import Scope
from ibis.expr.timecontext import canonicalize_context, localize_context

from .compiler import PySparkExprTranslator


class PySparkClient(SparkClient):
    """
    An ibis client that uses PySpark SQL Dataframe
    """

    def __init__(self, backend, session):
        super().__init__(backend, session)
        self.translator = PySparkExprTranslator()

    def compile(self, expr, timecontext=None, params=None, *args, **kwargs):
        """Compile an ibis expression to a PySpark DataFrame object
        """

        if timecontext is not None:
            session_timezone = self._session.conf.get(
                'spark.sql.session.timeZone'
            )
            # Since spark use session timezone for tz-naive timestamps
            # we localize tz-naive context here to match that behavior
            timecontext = localize_context(
                canonicalize_context(timecontext), session_timezone
            )

        # Insert params in scope
        if params is None:
            scope = Scope()
        else:
            scope = Scope(
                {param.op(): raw_value for param, raw_value in params.items()},
                timecontext,
            )
        return self.translator.translate(
            expr, scope=scope, timecontext=timecontext
        )

    def execute(
        self, expr, timecontext=None, params=None, limit='default', **kwargs
    ):
        if isinstance(expr, types.TableExpr):
            return self.compile(expr, timecontext, params, **kwargs).toPandas()
        elif isinstance(expr, types.ColumnExpr):
            # expression must be named for the projection
            expr = expr.name('tmp')
            return self.compile(
                expr.to_projection(), timecontext, params, **kwargs
            ).toPandas()['tmp']
        elif isinstance(expr, types.ScalarExpr):
            compiled = self.compile(expr, timecontext, params, **kwargs)
            if isinstance(compiled, Column):
                # attach result column to a fake DataFrame and
                # select the result
                compiled = self._session.range(0, 1).select(compiled)
            return compiled.toPandas().iloc[0, 0]
        else:
            raise com.IbisError(
                "Cannot execute expression of type: {}".format(type(expr))
            )
