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
from ibis.formats.pyarrow import dtype_from_pyarrow, infer_sequence_dtype

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


def schema_from_pandas_dataframe(
    df: pd.DataFrame, schema=None, inference_function=infer_sequence_dtype
):
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
                ibis_dtype = inference_function(pandas_column.values)
            else:
                ibis_dtype = dtype_from_pandas(pandas_dtype)

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


def schema_from_dask_dataframe(df, schema=None):
    # TODO(kszucs): we should limit the computation to the first partition or
    # even just the first row if we switch to `pa.infer_type()` in the inference
    # function
    return schema_from_pandas_dataframe(
        df,
        schema=schema,
        inference_function=lambda s: infer_sequence_dtype(s.compute()),
    )


def convert_pandas_dataframe(df, schema):
    if len(schema) != len(df.columns):
        raise ValueError("schema column count does not match input data column count")

    for column, dtype in zip(df.columns, schema.types):
        df[column] = convert_pandas_series(df[column], dtype)

    # return data with the schema's columns which may be different than the input columns
    df.columns = schema.names
    return df


def convert_pandas_series(s, dtype):
    pandas_type = dtype.to_pandas()

    if s.dtype == pandas_type and dtype.is_primitive():
        return s

    if dtype.is_boolean():
        if s.empty:
            return s.astype(pandas_type)
        elif pdt.is_object_dtype(s.dtype):
            return s
        elif s.dtype != pandas_type:
            return s.map(lambda value: pd.NA if pd.isna(value) else bool(value))
        else:
            return s
    elif dtype.is_timestamp():
        if pdt.is_datetime64tz_dtype(s.dtype):
            return s.dt.tz_convert(dtype.timezone)
        elif pdt.is_datetime64_dtype(s.dtype):
            return s.dt.tz_localize(dtype.timezone)
        else:
            try:
                return s.astype(pandas_type)
            except pd.errors.OutOfBoundsDatetime:  # uncovered
                try:
                    return s.map(date_parse)
                except TypeError:
                    return s
            except TypeError:
                try:
                    return pd.to_datetime(s).dt.tz_convert(dtype.timezone)
                except TypeError:
                    return pd.to_datetime(s).dt.tz_localize(dtype.timezone)
    elif dtype.is_date():
        if pdt.is_datetime64tz_dtype(s.dtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.astype(pandas_type, errors='ignore').dt.normalize()
    elif dtype.is_interval():
        try:
            return s.values.astype(pandas_type)
        except ValueError:  # can happen when `column` is DateOffsets  # uncovered
            return s
    elif dtype.is_string():
        return s.astype(pandas_type, errors='ignore')
    elif dtype.is_uuid():
        return s.map(lambda v: v if isinstance(v, UUID) else UUID(v))  # uncovered
    elif dtype.is_struct():

        def convert_element(values, names=dtype.names):
            if values is None or isinstance(values, dict) or pd.isna(values):
                return values
            return dict(zip(names, values))  # uncovered

        return s.map(convert_element)
    elif dtype.is_array():
        return s.map(lambda x: list(x) if util.is_iterable(x) else x)
    elif dtype.is_map():
        return s.map(lambda x: dict(x) if util.is_iterable(x) else x)
    elif dtype.is_json():

        def try_json(x):
            if x is None:
                return x
            try:
                return json.loads(x)
            except (TypeError, json.JSONDecodeError):
                return x

        return s.map(try_json).astype("object")
    else:
        # TODO(kszucs): the errors should be handled properly here
        try:
            return s.astype(pandas_type)
        except Exception:  # noqa: BLE001
            return s
