try:
    from ibis.bigquery.udf.api import udf  # noqa: F401
except ImportError:
    pass  # BigQuery UDFs are not supported in Python 2
