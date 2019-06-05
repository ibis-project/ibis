from ibis.spark.client import (  # noqa: F401; ImpalaDatabase,; ImpalaTable,
    SparkClient,
)


def connect():
    """Create a SparkClient for use with Ibis.

    Parameters
    ----------
    TODO

    Examples
    --------
    TODO

    Returns
    -------
    SparkClient
    """
    client = SparkClient()

    return client
