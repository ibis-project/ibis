import pyspark

from ibis.backends.base import BaseBackend

from .client import PySparkClient, PySparkTable
from .compiler import PySparkDatabaseTable


class Backend(BaseBackend):
    name = 'pyspark'
    client_class = PySparkClient
    table_class = PySparkDatabaseTable
    table_expr_class = PySparkTable

    def connect(self, session):
        """
        Create a `SparkClient` for use with Ibis.

        Pipes `**kwargs` into SparkClient, which pipes them into SparkContext.
        See documentation for SparkContext:
        https://spark.apache.org/docs/latest/api/python/_modules/pyspark/context.html#SparkContext
        """
        client = self.client_class(backend=self, session=session)

        # Spark internally stores timestamps as UTC values, and timestamp data
        # that is brought in without a specified time zone is converted as
        # local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
        client._session.conf.set('spark.sql.session.timeZone', 'UTC')

        return client

    @property
    def version(self):
        return pyspark.__version__

    @property
    def current_database(self):
        return self.client._catalog.currentDatabase()
