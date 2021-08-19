"""The dask client implementation."""

from __future__ import absolute_import

import re
from functools import partial
from typing import Dict, List, Mapping

import dask.dataframe as dd
import dateutil.parser
import numpy as np
import pandas as pd
import toolz
from dask.base import DaskMethodsMixin
from pandas.api.types import DatetimeTZDtype

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import Client, Database
from ibis.backends.pandas.client import (
    PANDAS_DATE_TYPES,
    PANDAS_STRING_TYPES,
    _inferable_pandas_dtypes,
    convert_timezone,
    ibis_dtype_to_pandas,
    ibis_schema_to_pandas,
)

from .core import execute_and_reset

infer_dask_dtype = pd.api.types.infer_dtype


_inferable_dask_dtypes = _inferable_pandas_dtypes


@sch.schema.register(dd.Series)
def schema_from_series(s):
    return sch.schema(tuple(s.iteritems()))


@sch.infer.register(dd.DataFrame)
def infer_dask_schema(df, schema=None):
    schema = schema if schema is not None else {}

    pairs = []
    for column_name, dask_dtype in df.dtypes.iteritems():
        if not isinstance(column_name, str):
            raise TypeError(
                'Column names must be strings to use the dask backend'
            )

        if column_name in schema:
            ibis_dtype = dt.dtype(schema[column_name])
        elif dask_dtype == np.object_:
            inferred_dtype = infer_dask_dtype(
                df[column_name].compute(), skipna=True
            )
            if inferred_dtype in {'mixed', 'decimal'}:
                # TODO: in principal we can handle decimal (added in pandas
                # 0.23)
                raise TypeError(
                    'Unable to infer type of column {0!r}. Try instantiating '
                    'your table from the client with client.table('
                    "'my_table', schema={{{0!r}: <explicit type>}})".format(
                        column_name
                    )
                )
            ibis_dtype = _inferable_dask_dtypes[inferred_dtype]
        else:
            ibis_dtype = dt.dtype(dask_dtype)

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


ibis_dtype_to_dask = ibis_dtype_to_pandas

ibis_schema_to_dask = ibis_schema_to_pandas


@sch.convert.register(DatetimeTZDtype, dt.Timestamp, dd.Series)
def convert_datetimetz_to_timestamp(in_dtype, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    return column.astype(out_dtype.to_dask())


DASK_STRING_TYPES = PANDAS_STRING_TYPES
DASK_DATE_TYPES = PANDAS_DATE_TYPES


@sch.convert.register(np.dtype, dt.Timestamp, dd.Series)
def convert_datetime64_to_timestamp(in_dtype, out_dtype, column):
    if in_dtype.type == np.datetime64:
        return column.astype(out_dtype.to_dask())
    try:
        # TODO - check this?
        series = pd.to_datetime(column, utc=True)
    except pd.errors.OutOfBoundsDatetime:
        inferred_dtype = infer_dask_dtype(column, skipna=True)
        if inferred_dtype in DASK_DATE_TYPES:
            # not great, but not really any other option
            return column.map(
                partial(convert_timezone, timezone=out_dtype.timezone)
            )
        if inferred_dtype not in DASK_STRING_TYPES:
            raise TypeError(
                (
                    'Conversion to timestamp not supported for Series of type '
                    '{!r}'
                ).format(inferred_dtype)
            )
        return column.map(dateutil.parser.parse)
    else:
        utc_dtype = DatetimeTZDtype('ns', 'UTC')
        return series.astype(utc_dtype).dt.tz_convert(out_dtype.timezone)


@sch.convert.register(np.dtype, dt.Interval, dd.Series)
def convert_any_to_interval(_, out_dtype, column):
    return column.values.astype(out_dtype.to_dask())


@sch.convert.register(np.dtype, dt.String, dd.Series)
def convert_any_to_string(_, out_dtype, column):
    result = column.astype(out_dtype.to_dask())
    return result


@sch.convert.register(np.dtype, dt.Boolean, dd.Series)
def convert_boolean_to_series(in_dtype, out_dtype, column):
    # XXX: this is a workaround until #1595 can be addressed
    in_dtype_type = in_dtype.type
    out_dtype_type = out_dtype.to_dask().type
    if in_dtype_type != np.object_ and in_dtype_type != out_dtype_type:
        return column.astype(out_dtype_type)
    return column


@sch.convert.register(object, dt.DataType, dd.Series)
def convert_any_to_any(_, out_dtype, column):
    return column.astype(out_dtype.to_dask())


dt.DataType.to_dask = ibis_dtype_to_dask
sch.Schema.to_dask = ibis_schema_to_dask


class DaskTable(ops.DatabaseTable):
    pass


class DaskDatabase(Database):
    pass


class DaskClient(Client):
    def __init__(self, backend, dictionary: Dict[str, dd.DataFrame]):
        self.backend = backend
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        self.dictionary = dictionary

    def table(self, name: str, schema: sch.Schema = None) -> DaskTable:
        df = self.dictionary[name]
        schema = sch.infer(df, schema=schema)
        return self.table_class(name, schema, self).to_expr()

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

    def list_tables(self, like: str = None) -> List[str]:
        """List the available tables."""
        tables = list(self.dictionary.keys())
        if like is not None:
            pattern = re.compile(like)
            return list(filter(lambda t: pattern.findall(t), tables))
        return tables

    def load_data(self, table_name: str, obj: dd.DataFrame, **kwargs):
        """Load data from `obj` into `table_name`.

        Parameters
        ----------
        table_name : str
        obj : dask.dataframe.DataFrame

        """
        # kwargs is a catch all for any options required by other backends.
        self.dictionary[table_name] = obj

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

    def get_schema(self, table_name: str, database: str = None) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name : str
            May be fully qualified
        database : str

        Returns
        -------
        ibis.expr.schema.Schema

        """
        return sch.infer(self.dictionary[table_name])
