"""The pandas client implementation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import toolz
from dateutil.parser import parse as date_parse
from pandas.api.types import CategoricalDtype, DatetimeTZDtype

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base import Database
from ibis.common.grounds import Immutable

_ibis_dtypes = toolz.valmap(
    np.dtype,
    {
        dt.Boolean: np.bool_,
        dt.Null: np.object_,
        dt.Array: np.object_,
        dt.String: np.object_,
        dt.Binary: np.object_,
        dt.Date: 'datetime64[ns]',
        dt.Time: 'timedelta64[ns]',
        dt.Timestamp: 'datetime64[ns]',
        dt.Int8: np.int8,
        dt.Int16: np.int16,
        dt.Int32: np.int32,
        dt.Int64: np.int64,
        dt.UInt8: np.uint8,
        dt.UInt16: np.uint16,
        dt.UInt32: np.uint32,
        dt.UInt64: np.uint64,
        dt.Float16: np.float16,
        dt.Float32: np.float32,
        dt.Float64: np.float64,
        dt.Decimal: np.object_,
        dt.Struct: np.object_,
    },
)


@dt.dtype.register(DatetimeTZDtype)
def from_pandas_tzdtype(value):
    return dt.Timestamp(timezone=str(value.tz))


@dt.dtype.register(CategoricalDtype)
def from_pandas_categorical(_):
    return dt.Category()


@dt.dtype.register(pd.core.arrays.string_.StringDtype)
def from_pandas_string(_):
    return dt.String()


@sch.schema.register(pd.Series)
def schema_from_series(s):
    return sch.schema(tuple(s.items()))


@sch.infer.register(pd.DataFrame)
def infer_pandas_schema(df: pd.DataFrame, schema=None):
    schema = schema if schema is not None else {}

    pairs = []
    for column_name in df.dtypes.keys():
        if not isinstance(column_name, str):
            raise TypeError('Column names must be strings to use the pandas backend')

        if column_name in schema:
            ibis_dtype = dt.dtype(schema[column_name])
        else:
            ibis_dtype = dt.infer(df[column_name]).value_type

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


def ibis_dtype_to_pandas(ibis_dtype: dt.DataType):
    """Convert ibis dtype to the pandas / numpy alternative."""
    assert isinstance(ibis_dtype, dt.DataType)

    if ibis_dtype.is_timestamp() and ibis_dtype.timezone:
        return DatetimeTZDtype('ns', ibis_dtype.timezone)
    elif ibis_dtype.is_interval():
        return np.dtype(f'timedelta64[{ibis_dtype.unit}]')
    elif ibis_dtype.is_category():
        return CategoricalDtype()
    else:
        return _ibis_dtypes.get(type(ibis_dtype), np.dtype(np.object_))


def ibis_schema_to_pandas(schema):
    return list(zip(schema.names, map(ibis_dtype_to_pandas, schema.types)))


@sch.convert.register(DatetimeTZDtype, dt.Timestamp, pd.Series)
def convert_datetimetz_to_timestamp(_, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    return column.astype(out_dtype.to_pandas(), errors='ignore')


PANDAS_STRING_TYPES = {'string', 'unicode', 'bytes'}
PANDAS_DATE_TYPES = {'datetime', 'datetime64', 'date'}


@sch.convert.register(np.dtype, dt.Interval, pd.Series)
def convert_any_to_interval(_, out_dtype, column):
    return column.values.astype(out_dtype.to_pandas())


@sch.convert.register(np.dtype, dt.String, pd.Series)
def convert_any_to_string(_, out_dtype, column):
    result = column.astype(out_dtype.to_pandas(), errors='ignore')
    return result


@sch.convert.register(np.dtype, dt.Boolean, pd.Series)
def convert_boolean_to_series(in_dtype, out_dtype, column):
    # XXX: this is a workaround until #1595 can be addressed
    in_dtype_type = in_dtype.type
    out_dtype_type = out_dtype.to_pandas().type
    if column.empty or (
        in_dtype_type != np.object_ and in_dtype_type != out_dtype_type
    ):
        return column.astype(out_dtype_type)
    return column


@sch.convert.register(DatetimeTZDtype, dt.Date, pd.Series)
def convert_timestamp_to_date(in_dtype, out_dtype, column):
    if in_dtype.tz is not None:
        column = column.dt.tz_convert("UTC").dt.tz_localize(None)
    return column.astype(out_dtype.to_pandas(), errors='ignore').dt.normalize()


@sch.convert.register(object, dt.DataType, pd.Series)
def convert_any_to_any(_, out_dtype, column):
    try:
        return column.astype(out_dtype.to_pandas())
    except pd.errors.OutOfBoundsDatetime:
        try:
            return column.map(date_parse)
        except TypeError:
            return column
    except Exception:  # noqa: BLE001
        return column


@sch.convert.register(object, dt.Struct, pd.Series)
def convert_struct_to_dict(_, out_dtype, column):
    def convert_element(values, names=out_dtype.names):
        if values is None or isinstance(values, dict) or pd.isna(values):
            return values
        return dict(zip(names, values))

    return column.map(convert_element)


@sch.convert.register(np.dtype, dt.Array, pd.Series)
def convert_array_to_series(in_dtype, out_dtype, column):
    return column.map(lambda x: x if x is None else list(x))


@sch.convert.register(np.dtype, dt.JSON, pd.Series)
def convert_json_to_series(in_, out, col: pd.Series):
    def try_json(x):
        if x is None:
            return x
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return x

    return pd.Series(list(map(try_json, col)), dtype="object")


class DataFrameProxy(Immutable, util.ToFrame):
    __slots__ = ('_df', '_hash')

    def __init__(self, df):
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_hash", hash((type(df), id(df))))

    def __hash__(self):
        return self._hash

    def __repr__(self):
        df_repr = util.indent(repr(self._df), spaces=2)
        return f"{self.__class__.__name__}:\n{df_repr}"

    def to_frame(self):
        return self._df


class PandasInMemoryTable(ops.InMemoryTable):
    data = rlz.instance_of(DataFrameProxy)


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass
