from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import dask
import dask.dataframe as dd
import pandas as pd
from dask.base import DaskMethodsMixin

# import the pandas execution module to register dispatched implementations of
# execute_node that the dask backend will later override
import ibis.backends.pandas.execution  # noqa: F401
import ibis.config
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.dask.client import (
    DaskDatabase,
    DaskTable,
    ibis_schema_to_dask,
)
from ibis.backends.dask.core import execute_and_reset
from ibis.backends.pandas import BasePandasBackend

# Make sure that the pandas backend options have been loaded
ibis.pandas


class Backend(BasePandasBackend):
    name = 'dask'
    database_class = DaskDatabase
    table_class = DaskTable
    backend_table_type = dd.DataFrame

    def do_connect(
        self,
        dictionary: MutableMapping[str, dd.DataFrame],
    ) -> None:
        """Construct a Dask backend client from a dictionary of data sources.

        Parameters
        ----------
        dictionary
            Mapping from `str` table names to Dask DataFrames.

        Examples
        --------
        >>> import ibis
        >>> data = {"t": "path/to/file.parquet", "s": "path/to/file.csv"}
        >>> ibis.dask.connect(data)
        """
        # register dispatchers
        from ibis.backends.dask import udf  # noqa: F401

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
        if limit != 'default' and limit is not None:
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

    @classmethod
    def _supports_conversion(cls, obj: Any) -> bool:
        return isinstance(obj, cls.backend_table_type)

    @staticmethod
    def _from_pandas(df: pd.DataFrame, npartitions: int = 1) -> dd.DataFrame:
        return dd.from_pandas(df, npartitions=npartitions)

    @staticmethod
    def _convert_schema(schema: sch.Schema):
        return ibis_schema_to_dask(schema)

    @classmethod
    def _convert_object(cls, obj: dd.DataFrame) -> dd.DataFrame:
        return obj

    def create_table(
        self,
        table_name: str,
        obj: dd.DataFrame | None = None,
        schema: sch.Schema | None = None,
    ):
        """Create a table."""
        super().create_table(table_name, obj=obj, schema=schema)
