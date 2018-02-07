import google.cloud.bigquery  # noqa: F401 fail early if bigquery is missing
import ibis.common as com
from ibis.config import options  # noqa: F401
from ibis.bigquery.client import BigQueryClient
from ibis.bigquery.compiler import dialect


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
    """
    Determine if expression can be successfully translated to execute on
    BigQuery
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(project_id, dataset_id):
    """Create a BigQueryClient for use with Ibis

    Parameters
    ----------
    project_id: str
    dataset_id: str

    Returns
    -------
    BigQueryClient
    """

    return BigQueryClient(project_id, dataset_id)
