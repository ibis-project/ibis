from typing import Mapping

import dask
import dask.dataframe as dd
import pandas as pd
import toolz
from dask.base import DaskMethodsMixin

# import the pandas execution module to register dispatched implementations of
# execute_node that the dask backend will later override
import ibis.backends.pandas.execution  # noqa: F401
import ibis.common.exceptions as com
import ibis.config
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.pandas import BasePandasBackend

from .client import DaskDatabase, DaskTable, ibis_schema_to_dask
from .core import execute_and_reset

# Make sure that the pandas backend options have been loaded
ibis.pandas


class Backend(BasePandasBackend):
    name = 'dask'
    database_class = DaskDatabase
    table_class = DaskTable

    def do_connect(self, dictionary):
        # register dispatchers
        from . import udf  # noqa: F401

        super().do_connect(dictionary)

    @property
    def version(self):
        return dask.__version__

    def execute(
        self,
        query: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: str = 'default',
        **kwargs,
    ):
        if limit != 'default':
            raise ValueError(
                'limit parameter to execute is not yet implemented in the '
                'dask backend'
            )

        if not isinstance(query, ir.Expr):
            raise TypeError(
                "`query` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(query).__name__
                )
            )

        result = self.compile(query, params, **kwargs)
        if isinstance(result, DaskMethodsMixin):
            return result.compute()
        else:
            return result

    def compile(
        self, query: ir.Expr, params: Mapping[ir.Expr, object] = None, **kwargs
    ):
        """Compile `expr`.

        Notes
        -----
        For the dask backend returns a dask graph that you can run ``.compute``
        on to get a pandas object.

        """
        return execute_and_reset(query, params=params, **kwargs)

    def create_table(
        self,
        table_name: str,
        obj: dd.DataFrame = None,
        schema: sch.Schema = None,
    ):
        """Create a table."""
        if obj is not None:
            df = obj
        elif schema is not None:
            dtypes = ibis_schema_to_dask(schema)
            df = schema.apply_to(
                dd.from_pandas(
                    pd.DataFrame(columns=list(map(toolz.first, dtypes))),
                    npartitions=1,
                )
            )
        else:
            raise com.IbisError('Must pass expr or schema')

        self.dictionary[table_name] = df
