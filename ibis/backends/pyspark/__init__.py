from ibis.backends.base import BaseBackend
from ibis.backends.spark.client import SparkDatabase, SparkQuery, SparkTable

from .client import PySparkClient, PySparkTable
from .compiler import PySparkDialect


class Backend(BaseBackend):
    name = 'pyspark'
    builder = None
    dialect = PySparkDialect
    database_class = SparkDatabase
    query_class = SparkQuery
    table_class = PySparkTable
    table_expr_class = SparkTable

    def connect(self, session):
        """
        Create a `SparkClient` for use with Ibis.

        Pipes `**kwargs` into SparkClient, which pipes them into SparkContext.
        See documentation for SparkContext:
        https://spark.apache.org/docs/latest/api/python/_modules/pyspark/context.html#SparkContext
        """
        client = PySparkClient(backend=self, session=session)

        # Spark internally stores timestamps as UTC values, and timestamp data
        # that is brought in without a specified time zone is converted as
        # local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
        client._session.conf.set('spark.sql.session.timeZone', 'UTC')

        return client
