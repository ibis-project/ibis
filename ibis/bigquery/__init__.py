from ibis.compat import PY2


if not PY2:
    from ibis.bigquery.udf.api import udf  # noqa: F401
