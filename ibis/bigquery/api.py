import ibis.common as com
from ibis.config import options  # noqa: F401
from ibis.bigquery.client import BigQueryClient


def compile(expr):
    """
    Force compilation of expression as though it were an expression depending
    on Impala. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from .compiler import to_sql
    return to_sql(expr)


def verify(expr):
    """
    Determine if expression can be successfully translated to execute on Impala
    """
    try:
        compile(expr)
        return True
    except com.TranslationError:
        return False


def connect(project_id, dataset_id):
    return BigQueryClient(project_id, dataset_id)
