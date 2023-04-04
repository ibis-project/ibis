"""The dask client implementation."""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dateutil.parser import parse as date_parse
from pandas.api.types import DatetimeTZDtype

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database
from ibis.backends.pandas.client import ibis_dtype_to_pandas, ibis_schema_to_pandas


@sch.schema.register(dd.Series)
def schema_from_series(s):
    return sch.schema(tuple(s.items()))


@sch.infer.register(dd.DataFrame)
def infer_dask_schema(df, schema=None):
    schema = schema if schema is not None else {}

    pairs = []
    for column_name, dask_dtype in df.dtypes.items():
        if not isinstance(column_name, str):
            raise TypeError('Column names must be strings to use the dask backend')

        if column_name in schema:
            ibis_dtype = dt.dtype(schema[column_name])
        elif dask_dtype == np.object_:
            # TODO: don't call compute here. ibis should just assume that
            # object dtypes are strings, which is what dask does. The user
            # can always explicitly pass in `schema=...` when creating a
            # table if they want to use a different dtype.
            ibis_dtype = dt.infer(df[column_name].compute()).value_type
        else:
            ibis_dtype = dt.dtype(dask_dtype)

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


ibis_dtype_to_dask = ibis_dtype_to_pandas

ibis_schema_to_dask = ibis_schema_to_pandas


@sch.convert.register(DatetimeTZDtype, dt.Timestamp, dd.Series)
def convert_datetimetz_to_timestamp(_, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    else:
        return column.dt.tz_localize(None)


@sch.convert.register(np.dtype, dt.Timestamp, dd.Series)
def convert_any_to_timestamp(_, out_dtype, column):
    if isinstance(dtype := out_dtype.to_dask(), DatetimeTZDtype):
        column = dd.to_datetime(column)
        timezone = out_dtype.timezone
        if getattr(column.dtype, "tz", None) is not None:
            return column.dt.tz_convert(timezone)
        else:
            return column.dt.tz_localize(timezone)
    else:
        try:
            return column.astype(dtype)
        except pd.errors.OutOfBoundsDatetime:
            try:
                return column.map(date_parse)
            except TypeError:
                return column


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
