from typing import Dict

from dask.dataframe import DataFrame

import ibis.config
from ibis.backends.base import BaseBackend
from ibis.backends.pandas import PandasExprTranslator

from . import udf  # noqa: F401,F403 - register dispatchers
from .client import DaskClient, DaskDatabase, DaskTable


class Backend(BaseBackend):
    name = 'dask'
    kind = 'pandas'
    builder = None
    translator = PandasExprTranslator
    database_class = DaskDatabase
    table_class = DaskTable

    def connect(self, dictionary: Dict[str, DataFrame]) -> DaskClient:
        """Construct a dask client from a dictionary of DataFrames.

        Parameters
        ----------
        dictionary : dict

        Returns
        -------
        DaskClient
        """
        return DaskClient(backend=self, dictionary=dictionary)

    def from_dataframe(
        self, df: DataFrame, name: str = 'df', client: DaskClient = None
    ) -> DaskTable:
        """
        convenience function to construct an ibis table
        from a DataFrame

        Parameters
        ----------
        df : DataFrame
        name : str, default 'df'
        client : Client, default new DaskClient
            client dictionary will be mutated with the
            name of the DataFrame

        Returns
        -------
        Table
        """

        if client is None:
            return self.connect({name: df}).table(name)
        client.dictionary[name] = df
        return client.table(name)

    def register_options(self):
        ibis.config.register_option(
            'enable_trace',
            False,
            'Whether enable tracing for dask execution. '
            'See ibis.dask.trace for details.',
            validator=ibis.config.is_bool,
        )
