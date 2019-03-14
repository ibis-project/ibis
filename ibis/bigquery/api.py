"""BigQuery public API."""

from typing import Optional

import google.cloud.bigquery  # noqa: F401, fail early if bigquery is missing
import google.auth.credentials
import pydata_google_auth

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
    """Compile an expression for BigQuery.

    Returns
    -------
    compiled : str

    See Also
    --------
    ibis.expr.types.Expr.compile

    """
    from ibis.bigquery.compiler import to_sql
    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """Check if an expression can be compiled using BigQuery."""
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


SCOPES = ["https://www.googleapis.com/auth/bigquery"]
CLIENT_ID = (
    "546535678771-gvffde27nd83kfl6qbrnletqvkdmsese.apps.googleusercontent.com"
)
CLIENT_SECRET = "iU5ohAF2qcqrujegE3hQ1cPt"


def connect(
    project_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    credentials: Optional[google.auth.credentials.Credentials] = None,
) -> BigQueryClient:
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
    if credentials is None:
        credentials, project_id = pydata_google_auth.default(
            SCOPES,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )

    return BigQueryClient(
        project_id, dataset_id=dataset_id, credentials=credentials
    )
