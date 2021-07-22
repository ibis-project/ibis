import ibis.config
from ibis.backends.pandas import BasePandasBackend

from . import udf  # noqa: F401,F403 - register dispatchers
from .client import DaskClient, DaskDatabase, DaskTable
from .execution import execute  # noqa F401

# Make sure that the pandas backend is loaded, dispatching has been
# executed, and options have been loaded
ibis.pandas


class Backend(BasePandasBackend):
    name = 'dask'
    database_class = DaskDatabase
    table_class = DaskTable
    client_class = DaskClient
