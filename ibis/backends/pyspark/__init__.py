from ibis.backends.spark import BaseSparkBackend

from .client import PySparkClient
from .compiler import PySparkTable


class Backend(BaseSparkBackend):
    name = 'pyspark'
    client_class = PySparkClient
    table_class = PySparkTable
