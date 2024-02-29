from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pandas.api.types as pdt

import ibis.expr.datatypes as dt
from ibis.backends.pandas.convert import PandasConverter
from ibis.formats.pandas import DataMapper, PandasType


class DaskConverter(DataMapper):
    @classmethod
    def convert_scalar(cls, obj, dtype):
        return PandasConverter.convert_scalar(obj, dtype)

    @classmethod
    def convert_column(cls, obj, dtype):
        pandas_type = PandasType.from_ibis(dtype)

        method_name = f"convert_{dtype.__class__.__name__}"
        convert_method = getattr(cls, method_name, cls.convert_default)

        return convert_method(obj, dtype, pandas_type)

    @classmethod
    def convert_default(cls, s, dtype, pandas_type):
        if pandas_type == object:
            func = lambda x: x if x is pd.NA else dt.normalize(dtype, x)
            meta = (s.name, pandas_type)
            return s.map(func, na_action="ignore", meta=meta).astype(pandas_type)
        else:
            return s.astype(pandas_type)

    @classmethod
    def convert_Integer(cls, s, dtype, pandas_type):
        if pdt.is_datetime64_any_dtype(s.dtype):
            return s.astype("int64").floordiv(int(1e9)).astype(pandas_type)
        else:
            return s.astype(pandas_type)

    convert_SignedInteger = convert_UnsignedInteger = convert_Integer
    convert_Int64 = convert_Int32 = convert_Int16 = convert_Int8 = convert_SignedInteger
    convert_UInt64 = convert_UInt32 = convert_UInt16 = convert_UInt8 = (
        convert_UnsignedInteger
    )

    @classmethod
    def convert_Floating(cls, s, dtype, pandas_type):
        if pdt.is_datetime64_any_dtype(s.dtype):
            return s.astype("int64").floordiv(int(1e9)).astype(pandas_type)
        else:
            return s.astype(pandas_type)

    convert_Float64 = convert_Float32 = convert_Float16 = convert_Floating

    @classmethod
    def convert_Timestamp(cls, s, dtype, pandas_type):
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            return s.dt.tz_convert(dtype.timezone)
        elif pdt.is_datetime64_dtype(s.dtype):
            return s.dt.tz_localize(dtype.timezone)
        elif pdt.is_numeric_dtype(s.dtype):
            return dd.to_datetime(s, unit="s").dt.tz_localize(dtype.timezone)
        else:
            return dd.to_datetime(s, utc=True).dt.tz_localize(dtype.timezone)

    @classmethod
    def convert_Date(cls, s, dtype, pandas_type):
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif pdt.is_numeric_dtype(s.dtype):
            s = dd.to_datetime(s, unit="D")
        else:
            s = dd.to_datetime(s)

        return s.dt.normalize()

    @classmethod
    def convert_String(cls, s, dtype, pandas_type):
        # TODO(kszucs): should switch to the new pandas string type and convert
        # object columns using s.convert_dtypes() method
        return s.map(str, na_action="ignore").astype(object)
