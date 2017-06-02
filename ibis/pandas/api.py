from ibis.pandas.client import PandasClient
from ibis.pandas.execution import execute  # noqa: F401


def connect(dictionary):
    return PandasClient(dictionary)
