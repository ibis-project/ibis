"""The pandas client implementation."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pandas as pd
import toolz
from dateutil.parser import parse as date_parse
from pandas.api.types import CategoricalDtype, DatetimeTZDtype

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base import Database
from ibis.expr.operations.relations import TableProxy

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

if TYPE_CHECKING:
    import pyarrow as pa


@dt.dtype.register(DatetimeTZDtype)
def from_pandas_tzdtype(value):
    return dt.Timestamp(timezone=str(value.tz))


@dt.dtype.register(CategoricalDtype)
def from_pandas_categorical(_):
    return dt.String()


@dt.dtype.register(pd.core.dtypes.base.ExtensionDtype)
def from_pandas_extension_dtype(t):
    return getattr(dt, t.__class__.__name__.replace("Dtype", "").lower())


try:
    _arrow_dtype_class = pd.ArrowDtype
except AttributeError:
    warnings.warn(
        f"The `ArrowDtype` class is not available in pandas {pd.__version__}. "
        "Install pandas >= 1.5.0 for interop with pandas and arrow dtype support"
    )
else:

    @dt.dtype.register(_arrow_dtype_class)
    def from_pandas_arrow_extension_dtype(t):
        import ibis.backends.pyarrow.datatypes as _  # noqa: F401

        return dt.dtype(t.pyarrow_dtype)


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
    else:
        return _ibis_dtypes.get(type(ibis_dtype), np.dtype(np.object_))


def ibis_schema_to_pandas(schema):
    return list(zip(schema.names, map(ibis_dtype_to_pandas, schema.types)))


@sch.convert.register(DatetimeTZDtype, dt.Timestamp, pd.Series)
def convert_datetimetz_to_timestamp(_, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    else:
        return column.dt.tz_localize(None)


@sch.convert.register(np.dtype, dt.Interval, pd.Series)
def convert_any_to_interval(_, out_dtype, column):
    return column.values.astype(out_dtype.to_pandas())


@sch.convert.register(np.dtype, dt.String, pd.Series)
def convert_any_to_string(_, out_dtype, column):
    result = column.astype(out_dtype.to_pandas(), errors='ignore')
    return result


@sch.convert.register(np.dtype, dt.UUID, pd.Series)
def convert_any_to_uuid(_, out_dtype, column):
    return column.map(lambda v: v if isinstance(v, UUID) else UUID(v))


@sch.convert.register(np.dtype, dt.Boolean, pd.Series)
def convert_boolean_to_series(in_dtype, out_dtype, column):
    # XXX: this is a workaround until #1595 can be addressed
    in_dtype_type = in_dtype.type
    out_dtype_type = out_dtype.to_pandas().type
    if column.empty:
        return column.astype(out_dtype_type)
    elif in_dtype_type != np.object_ and in_dtype_type != out_dtype_type:
        return column.map(lambda value: pd.NA if pd.isna(value) else bool(value))
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
    except Exception:  # noqa: BLE001
        return column


@sch.convert.register(np.dtype, dt.Timestamp, pd.Series)
def convert_any_to_timestamp(_, out_dtype, column):
    try:
        return column.astype(out_dtype.to_pandas())
    except pd.errors.OutOfBoundsDatetime:
        try:
            return column.map(date_parse)
        except TypeError:
            return column
    except TypeError:
        column = pd.to_datetime(column)
        timezone = out_dtype.timezone
        try:
            return column.dt.tz_convert(timezone)
        except TypeError:
            return column.dt.tz_localize(timezone)


@sch.convert.register(object, dt.Struct, pd.Series)
def convert_struct_to_dict(_, out_dtype, column):
    def convert_element(values, names=out_dtype.names):
        if values is None or isinstance(values, dict) or pd.isna(values):
            return values
        return dict(zip(names, values))

    return column.map(convert_element)


@sch.convert.register(np.dtype, dt.Array, pd.Series)
def convert_array_to_series(in_dtype, out_dtype, column):
    return column.map(lambda x: list(x) if util.is_iterable(x) else x)


@sch.convert.register(np.dtype, dt.Map, pd.Series)
def convert_map_to_series(in_dtype, out_dtype, column):
    return column.map(lambda x: dict(x) if util.is_iterable(x) else x)


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


class DataFrameProxy(TableProxy):
    __slots__ = ()

    def to_frame(self) -> pd.DataFrame:
        return self._data

    def to_pyarrow(self, schema: sch.Schema) -> pa.Table:
        import pyarrow as pa

        from ibis.backends.pyarrow.datatypes import ibis_to_pyarrow_schema

        return pa.Table.from_pandas(self._data, schema=ibis_to_pyarrow_schema(schema))


class PandasTable(ops.DatabaseTable):
    pass


class PandasDatabase(Database):
    pass
