from ibis.spark.client import (  # noqa: F401; ImpalaDatabase,; ImpalaTable,
    SparkClient,
)


def connect(**kwargs):
    """
    Create a `SparkClient` for use with Ibis. Pipes **kwargs into SparkClient,
    which pipes them into SparkContext. See documentation for SparkContext:
    https://spark.apache.org/docs/latest/api/python/_modules/pyspark/context.html#SparkContext
    """
    client = SparkClient(**kwargs)

    return client
