from ibis.bigquery.client import (BigQueryClient)
from ibis.config import options
import ibis.common as com


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
