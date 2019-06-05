from ibis.spark.client import (  # noqa: F401
    SparkClient,
    #ImpalaDatabase,
    #ImpalaTable,
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
    params = {}

    client = SparkClient()
    # TODO
    # else:
    #     if options.default_backend is None:
    #         options.default_backend = client

    return client