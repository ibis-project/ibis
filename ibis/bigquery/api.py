"""BigQuery public API."""

import google.cloud.bigquery  # noqa: F401 fail early if bigquery is missing
import ibis.common as com

from ibis.config import options  # noqa: F401
from ibis.bigquery.client import BigQueryClient
from ibis.bigquery.compiler import dialect

try:
    from ibis.bigquery.udf import udf  # noqa: F401
except ImportError:
    pass


__all__ = (
    'compile',
    'connect',
    'verify',
    'udf',
)


def compile(expr, params=None):
    """
    Force compilation of expression as though it were an expression depending
    on BigQuery. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from ibis.bigquery.compiler import to_sql
    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """Check if an expression can be successfully translated using BigQuery."""
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(project_id, dataset_id, credentials=None):
    """Create a BigQueryClient for use with Ibis.

    Parameters
    ----------
    project_id : str
        A BigQuery project id.
    dataset_id : str
        A dataset id that lives inside of the project indicated by
        `project_id`.
    credentials : google.auth.credentials.Credentials

    Returns
    -------
    BigQueryClient

    """
    import pydata_google_auth

    if credentials is None:
        credentials, project_id = pydata_google_auth.default()

    return BigQueryClient(project_id, dataset_id, credentials=credentials)
