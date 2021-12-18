import pandas as pd
import toolz

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend

from .client import PandasDatabase, PandasTable, ibis_schema_to_pandas


class BasePandasBackend(BaseBackend):
    """
    Base class for backends based on pandas.
    """

    def do_connect(self, dictionary):
        """Construct a client from a dictionary of DataFrames.

        Parameters
        ----------
        dictionary : dict

        Returns
        -------
        Backend
        """
        # register dispatchers
        from . import execution  # noqa F401
        from . import udf  # noqa F401

        self.dictionary = dictionary

    def from_dataframe(self, df, name='df', client=None):
        """
        convenience function to construct an ibis table
        from a DataFrame

        Parameters
        ----------
        df : DataFrame
        name : str, default 'df'
        client : Backend, optional
            client dictionary will be mutated with the name of the DataFrame,
            if not provided a new client is created

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
            f'Whether enable tracing for {self.name} execution. '
            'See ibis.{self.name}.trace for details.',
            validator=ibis.config.is_bool,
        )

    @property
    def version(self) -> str:
        return pd.__version__

    @property
    def current_database(self):
        raise NotImplementedError('pandas backend does not support databases')

    def list_databases(self, like=None):
        raise NotImplementedError('pandas backend does not support databases')

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self.dictionary.keys()), like)

    def table(self, name: str, schema: sch.Schema = None):
        df = self.dictionary[name]
        schema = sch.infer(df, schema=schema)
        return self.table_class(name, schema, self).to_expr()

    def database(self, name=None):
        return self.database_class(name, self)

    def load_data(self, table_name, obj, **kwargs):
        # kwargs is a catch all for any options required by other backends.
        self.dictionary[table_name] = obj

    def get_schema(self, table_name, database=None):
        return sch.infer(self.dictionary[table_name])

    def compile(self, expr, *args, **kwargs):
        return expr


class Backend(BasePandasBackend):
    name = 'pandas'
    database_class = PandasDatabase
    table_class = PandasTable

    def execute(self, query, params=None, limit='default', **kwargs):
        from .core import execute_and_reset

        if limit != 'default':
            raise ValueError(
                'limit parameter to execute is not yet implemented in the '
                'pandas backend'
            )

        if not isinstance(query, ir.Expr):
            raise TypeError(
                "`query` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(query).__name__
                )
            )
        return execute_and_reset(query, params=params, **kwargs)

    def create_table(self, table_name, obj=None, schema=None):
        """Create a table."""
        if obj is None and schema is None:
            raise com.IbisError('Must pass expr or schema')

        if obj is not None:
            df = pd.DataFrame(obj)
        else:
            dtypes = ibis_schema_to_pandas(schema)
            df = schema.apply_to(
                pd.DataFrame(columns=list(map(toolz.first, dtypes)))
            )

        self.dictionary[table_name] = df
