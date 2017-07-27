from __future__ import absolute_import

from ibis.pandas.client import PandasClient
from ibis.pandas.decimal import execute_node  # noqa: F401
from ibis.pandas.execution import execute  # noqa: F401


def connect(dictionary):
    return PandasClient(dictionary)
