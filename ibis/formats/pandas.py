from __future__ import annotations

import json
import warnings
from uuid import UUID

import numpy as np
import pandas as pd
import pandas.api.types as pdt
from dateutil.parser import parse as date_parse

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import util
from ibis.formats.numpy import dtype_from_numpy, dtype_to_numpy
from ibis.formats.pyarrow import _infer_array_dtype, dtype_from_pyarrow

_has_arrow_dtype = hasattr(pd, "ArrowDtype")

if not _has_arrow_dtype:
    warnings.warn(
        f"The `ArrowDtype` class is not available in pandas {pd.__version__}. "
        "Install pandas >= 1.5.0 for interop with pandas and arrow dtype support"
    )


def dtype_to_pandas(dtype: dt.DataType):
    """Convert ibis dtype to the pandas / numpy alternative."""
    assert isinstance(dtype, dt.DataType)

    if dtype.is_timestamp() and dtype.timezone:
        return pdt.DatetimeTZDtype('ns', dtype.timezone)
    elif dtype.is_interval():
        return np.dtype(f'timedelta64[{dtype.unit.short}]')
    else:
        return dtype_to_numpy(dtype)


def dtype_from_pandas(typ, nullable=True):
    if pdt.is_datetime64tz_dtype(typ):
        return dt.Timestamp(timezone=str(typ.tz), nullable=nullable)
    elif pdt.is_datetime64_dtype(typ):
        return dt.Timestamp(nullable=nullable)
    elif pdt.is_categorical_dtype(typ):
        return dt.String(nullable=nullable)
    elif pdt.is_extension_array_dtype(typ):
        if _has_arrow_dtype and isinstance(typ, pd.ArrowDtype):
            return dtype_from_pyarrow(typ.pyarrow_dtype, nullable=nullable)
        else:
            name = typ.__class__.__name__.replace("Dtype", "")
            klass = getattr(dt, name)
            return klass(nullable=nullable)
    else:
        return dtype_from_numpy(typ, nullable=nullable)


def schema_to_pandas(schema):
    pandas_types = map(dtype_to_pandas, schema.types)
    return list(zip(schema.names, pandas_types))


def schema_from_pandas(schema):
    ibis_types = {name: dtype_from_pandas(typ) for name, typ in schema}
    return sch.schema(ibis_types)


def schema_from_pandas_dataframe(df: pd.DataFrame, schema=None):
    schema = schema if schema is not None else {}

    pairs = []
    for column_name in df.dtypes.keys():
        if not isinstance(column_name, str):
            raise TypeError('Column names must be strings to use the pandas backend')

        if column_name in schema:
            ibis_dtype = schema[column_name]
        else:
            pandas_column = df[column_name]
            pandas_dtype = pandas_column.dtype
            if pandas_dtype == np.object_:
                ibis_dtype = _infer_array_dtype(pandas_column.values)
            else:
                ibis_dtype = dtype_from_pandas(pandas_dtype)

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


@sch.convert.register(pdt.DatetimeTZDtype, dt.Timestamp, pd.Series)
def convert_datetimetz_to_timestamp(_, out_dtype, column):
    output_timezone = out_dtype.timezone
    if output_timezone is not None:
        return column.dt.tz_convert(output_timezone)
    else:
        return column.dt.tz_localize(None)


@sch.convert.register(np.dtype, dt.Interval, pd.Series)
def convert_any_to_interval(_, out_dtype, column):
    values = column.values
    pandas_dtype = out_dtype.to_pandas()
    try:
        return values.astype(pandas_dtype)
    except ValueError:  # can happen when `column` is DateOffsets
        return column


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


@sch.convert.register(pdt.DatetimeTZDtype, dt.Date, pd.Series)
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
