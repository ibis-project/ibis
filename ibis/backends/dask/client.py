"""The dask client implementation."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database
from ibis.backends.pandas.client import (
    PANDAS_DATE_TYPES,
    PANDAS_STRING_TYPES,
    _inferable_pandas_dtypes,
    ibis_dtype_to_pandas,
    ibis_schema_to_pandas,
)

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
