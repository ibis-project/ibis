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
from multipledispatch import Dispatcher
from pandas.api.types import DatetimeTZDtype
from pkg_resources import parse_version

import ibis.client as client
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.pandas.client import (  # noqa: F401
    PANDAS_DATE_TYPES,
    PANDAS_STRING_TYPES,
    _ibis_dtypes,
    _inferable_pandas_dtypes,
    _numpy_dtypes,
    convert_timezone,
    from_numpy_dtype,
    from_pandas_categorical,
    from_pandas_tzdtype,
    ibis_dtype_to_pandas,
    ibis_schema_to_pandas,
    infer_array,
    infer_numpy_scalar,
    infer_pandas_timestamp,
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

convert = Dispatcher(
    'convert',
    doc="""\
Convert `column` to the dask dtype corresponding to `out_dtype`, where the
dtype of `column` is `in_dtype`.

Parameters
----------
in_dtype : Union[np.dtype, dask_dtype]
    The dtype of `column`, used for dispatching
out_dtype : ibis.expr.datatypes.DataType
    The requested ibis type of the output
column : dd.Series
    The column to convert

Returns
-------
result : dd.Series
    The converted column
""",
)


@convert.register(DatetimeTZDtype, dt.Timestamp, dd.Series)
def convert_datetimetz_to_timestamp(in_dtype, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    return column.astype(out_dtype.to_dask(), errors='ignore')


DASK_STRING_TYPES = PANDAS_STRING_TYPES
DASK_DATE_TYPES = PANDAS_DATE_TYPES


@convert.register(np.dtype, dt.Timestamp, dd.Series)
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


@convert.register(np.dtype, dt.Interval, dd.Series)
def convert_any_to_interval(_, out_dtype, column):
    return column.values.astype(out_dtype.to_dask())


@convert.register(np.dtype, dt.String, dd.Series)
def convert_any_to_string(_, out_dtype, column):
    result = column.astype(out_dtype.to_dask(), errors='ignore')
    return result


@convert.register(np.dtype, dt.Boolean, dd.Series)
def convert_boolean_to_series(in_dtype, out_dtype, column):
    # XXX: this is a workaround until #1595 can be addressed
    in_dtype_type = in_dtype.type
    out_dtype_type = out_dtype.to_dask().type
    if in_dtype_type != np.object_ and in_dtype_type != out_dtype_type:
        return column.astype(out_dtype_type)
    return column


@convert.register(object, dt.DataType, dd.Series)
def convert_any_to_any(_, out_dtype, column):
    return column.astype(out_dtype.to_dask(), errors='ignore')


def ibis_schema_apply_to(schema: sch.Schema, df: dd.DataFrame) -> dd.DataFrame:
    """Applies the Ibis schema to a dask DataFrame

    Parameters
    ----------
    schema : ibis.schema.Schema
    df : dask.dataframe.DataFrame

    Returns
    -------
    df : dask.dataframeDataFrame

    Notes
    -----
    Mutates `df`
    """

    for column, dtype in schema.items():
        dask_dtype = dtype.to_dask()
        col = df[column]
        col_dtype = col.dtype

        try:
            not_equal = dask_dtype != col_dtype
        except TypeError:
            # ugh, we can't compare dtypes coming from dask, assume not equal
            not_equal = True

        if not_equal or isinstance(dtype, dt.String):
            df[column] = convert(col_dtype, dtype, col)

    return df


dt.DataType.to_dask = ibis_dtype_to_dask
sch.Schema.to_dask = ibis_schema_to_dask
sch.Schema.apply_to = ibis_schema_apply_to


class DaskTable(ops.DatabaseTable):
    pass


class DaskDatabase(client.Database):
    pass


class DaskClient(client.Client):

    dialect = None  # defined in ibis.dask.api

    def __init__(self, dictionary: Dict[str, dd.DataFrame]):
        self.dictionary = dictionary

    def table(self, name: str, schema: sch.Schema = None) -> DaskTable:
        df = self.dictionary[name]
        schema = sch.infer(df, schema=schema)
        return DaskTable(name, schema, self).to_expr()

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
        return execute_and_reset(query, params=params, **kwargs)

    def compile(self, expr: ir.Expr, *args, **kwargs):
        """Compile `expr`.

        Notes
        -----
        For the dask backend this is a no-op.

        """
        return expr

    def database(self, name: str = None) -> DaskDatabase:
        """Construct a database called `name`."""
        return DaskDatabase(name, self)

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
        if obj is None and schema is None:
            raise com.IbisError('Must pass expr or schema')

        if obj is not None:
            df = obj
        else:
            # TODO - this isn't right
            dtypes = ibis_schema_to_dask(schema)
            df = schema.apply_to(
                dd.from_pandas(
                    pd.DataFrame(columns=list(map(toolz.first, dtypes))),
                    npartitions=1,
                )
            )

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

    def exists_table(self, name: str) -> bool:
        """Determine if the indicated table or view exists.

        Parameters
        ----------
        name : str
        database : str

        Returns
        -------
        bool

        """
        return bool(self.list_tables(like=name))

    @property
    def version(self) -> str:
        """Return the version of the underlying backend library."""
        return parse_version(dd.__version__)
