from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask
import dask.dataframe as dd
import pandas as pd
from dask.base import DaskMethodsMixin

# import the pandas execution module to register dispatched implementations of
# execute_node that the dask backend will later override
import ibis.backends.pandas.execution
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.dask.core import execute_and_reset
from ibis.backends.pandas import BasePandasBackend
from ibis.backends.pandas.core import _apply_schema
from ibis.formats.pandas import DaskData

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

# Make sure that the pandas backend options have been loaded
ibis.pandas  # noqa: B018


class Backend(BasePandasBackend):
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
        # register dispatchers
        from ibis.backends.dask import udf  # noqa: F401

        if dictionary is None:
            dictionary = {}

        for k, v in dictionary.items():
            if not isinstance(v, (dd.DataFrame, pd.DataFrame)):
                raise TypeError(
                    f"Expected an instance of 'dask.dataframe.DataFrame' for {k!r},"
                    f" got an instance of '{type(v).__name__}' instead."
                )
        super().do_connect(dictionary)

    @property
    def version(self):
        return dask.__version__

    def execute(
        self,
        query: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: str = "default",
        **kwargs,
    ):
        if limit != "default" and limit is not None:
            raise ValueError(
                "limit parameter to execute is not yet implemented in the "
                "dask backend"
            )

        if not isinstance(query, ir.Expr):
            raise TypeError(
                "`query` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(query).__name__
                )
            )

        compiled = self.compile(query, params, **kwargs)
        if isinstance(compiled, DaskMethodsMixin):
            result = compiled.compute()
        else:
            result = compiled
        return _apply_schema(query.op(), result)

    def compile(
        self, query: ir.Expr, params: Mapping[ir.Expr, object] | None = None, **kwargs
    ):
        """Compile `expr`.

        Returns
        -------
        dask.dataframe.core.DataFrame | dask.dataframe.core.Series | dask.dataframe.core.Scalar
            Dask graph.
        """
        params = {
            k.op() if isinstance(k, ir.Expr) else k: v
            for k, v in ({} if params is None else params).items()
        }

        return execute_and_reset(query.op(), params=params, **kwargs)

    def table(self, name: str, schema: sch.Schema | None = None):
        df = self.dictionary[name]
        schema = schema or self.schemas.get(name, None)
        schema = DaskData.infer_table(df, schema=schema)
        return ops.DatabaseTable(name, schema, self).to_expr()

    @classmethod
    def _supports_conversion(cls, obj: Any) -> bool:
        return isinstance(obj, cls.backend_table_type)

    @staticmethod
    def _from_pandas(df: pd.DataFrame, npartitions: int = 1) -> dd.DataFrame:
        return dd.from_pandas(df, npartitions=npartitions)

    @classmethod
    def _convert_object(cls, obj: dd.DataFrame) -> dd.DataFrame:
        return obj

    def _load_into_cache(self, name, expr):
        self.create_table(name, self.compile(expr).persist())
