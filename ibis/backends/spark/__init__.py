"""Spark backend."""
import ibis.common.exceptions as com

from .client import SparkClient
from .compiler import dialect  # noqa: F401
from .udf import udf  # noqa: F401


def compile(expr, params=None):
    """Force compilation of expression.

    Returns
    -------
    str

    """
    from .compiler import to_sql

    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """
    Determine if expression can be successfully translated to execute on Impala
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(spark_session):
    """
    Create a `SparkClient` for use with Ibis.

    Pipes `**kwargs` into SparkClient, which pipes them into SparkContext.
    See documentation for SparkContext:
    https://spark.apache.org/docs/latest/api/python/_modules/pyspark/context.html#SparkContext
    """
    client = SparkClient(spark_session)

    # Spark internally stores timestamps as UTC values, and timestamp data that
    # is brought in without a specified time zone is converted as local time to
    # UTC with microsecond resolution.
    # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
    client._session.conf.set('spark.sql.session.timeZone', 'UTC')

    return client
