from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask
import dask.dataframe as dd
import pandas as pd

import ibis.common.exceptions as com

# import the pandas execution module to register dispatched implementations of
# execute_node that the dask backend will later override
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import NoUrl
from ibis.backends.pandas import BasePandasBackend
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping, MutableMapping


class Backend(BasePandasBackend, NoUrl):
    name = "dask"
    backend_table_type = dd.DataFrame
    supports_in_memory_tables = False

    def do_connect(
        self,
        dictionary: MutableMapping[str, dd.DataFrame] | None = None,
    ) -> None:
        """Construct a Dask backend client from a dictionary of data sources.

        Parameters
        ----------
        dictionary
            An optional mapping from `str` table names to Dask DataFrames.

        Examples
        --------
        >>> import ibis
        >>> import dask.dataframe as dd
        >>> data = {
        ...     "t": dd.read_parquet("path/to/file.parquet"),
        ...     "s": dd.read_csv("path/to/file.csv"),
        ... }
        >>> ibis.dask.connect(data)

        """
        if dictionary is None:
            dictionary = {}

        for k, v in dictionary.items():
            if not isinstance(v, (dd.DataFrame, pd.DataFrame)):
                raise TypeError(
                    f"Expected an instance of 'dask.dataframe.DataFrame' for {k!r},"
                    f" got an instance of '{type(v).__name__}' instead."
                )
        super().do_connect(dictionary)

    def disconnect(self) -> None:
        pass

    @property
    def version(self):
        return dask.__version__

    def _validate_args(self, expr, limit, timecontext):
        if timecontext is not None:
            raise com.UnsupportedArgumentError(
                "The Dask backend does not support timecontext"
            )
        if limit != "default" and limit is not None:
            raise com.UnsupportedArgumentError(
                "limit parameter to execute is not yet implemented in the "
                "dask backend"
            )
        if not isinstance(expr, ir.Expr):
            raise TypeError(
                "`expr` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(expr).__name__
                )
            )

    def compile(
        self,
        expr: ir.Expr,
        params: dict | None = None,
        limit: int | None = None,
        timecontext=None,
    ):
        from ibis.backends.dask.executor import DaskExecutor

        self._validate_args(expr, limit, timecontext)
        params = params or {}
        params = {k.op() if isinstance(k, ir.Expr) else k: v for k, v in params.items()}

        return DaskExecutor.compile(expr.op(), backend=self, params=params)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: str = "default",
        timecontext=None,
        **kwargs,
    ):
        from ibis.backends.dask.executor import DaskExecutor

        self._validate_args(expr, limit, timecontext)
        params = params or {}
        params = {k.op() if isinstance(k, ir.Expr) else k: v for k, v in params.items()}

        return DaskExecutor.execute(expr.op(), backend=self, params=params)

    def read_csv(
        self, source: str | pathlib.Path, table_name: str | None = None, **kwargs: Any
    ):
        """Register a CSV file as a table in the current session.

        Parameters
        ----------
        source
            The data source. Can be a local or remote file, pathlike objects
            also accepted.
        table_name
            An optional name to use for the created table. This defaults to
            a generated name.
        **kwargs
            Additional keyword arguments passed to Pandas loading function.
            See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or util.gen_name("read_csv")
        df = dd.read_csv(source, **kwargs)
        self.dictionary[table_name] = df
        return self.table(table_name)

    def read_parquet(
        self, source: str | pathlib.Path, table_name: str | None = None, **kwargs: Any
    ):
        """Register a parquet file as a table in the current session.

        Parameters
        ----------
        source
            The data source(s). May be a path to a file, an iterable of files,
            or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a generated name.
        **kwargs
            Additional keyword arguments passed to Pandas loading function.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or util.gen_name("read_parquet")
        df = dd.read_parquet(source, **kwargs)
        self.dictionary[table_name] = df
        return self.table(table_name)

    def table(self, name: str, schema: sch.Schema | None = None):
        df = self.dictionary[name]
        schema = schema or self.schemas.get(name, None)
        schema = PandasData.infer_table(df.head(1), schema=schema)
        return ops.DatabaseTable(name, schema, self).to_expr()

    def _convert_object(self, obj) -> dd.DataFrame:
        if isinstance(obj, dd.DataFrame):
            return obj

        pandas_df = super()._convert_object(obj)
        return dd.from_pandas(pandas_df, npartitions=1)

    def _load_into_cache(self, name, expr):
        self.create_table(name, self.compile(expr).persist())
