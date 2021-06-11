import ibis.config
from ibis.backends.base import BaseBackend

from .client import PandasClient, PandasDatabase, PandasTable
from .execution import execute
from .udf import udf  # noqa F401


class Backend(BaseBackend):
    name = 'pandas'
    kind = 'pandas'
    database_class = PandasDatabase
    table_class = PandasTable

    def connect(self, dictionary):
        """Construct a pandas client from a dictionary of DataFrames.

        Parameters
        ----------
        dictionary : dict

        Returns
        -------
        PandasClient
        """
        return PandasClient(backend=self, dictionary=dictionary)

    def execute(self, *args, **kwargs):
        return execute(*args, **kwargs)

    def from_dataframe(self, df, name='df', client=None):
        """
        convenience function to construct an ibis table
        from a DataFrame

        Parameters
        ----------
        df : DataFrame
        name : str, default 'df'
        client : Client, default new PandasClient
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
            'Whether enable tracing for pandas execution. '
            'See ibis.pandas.trace for details.',
            validator=ibis.config.is_bool,
        )
