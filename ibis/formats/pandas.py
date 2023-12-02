from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pandas.api.types as pdt

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import DataMapper, SchemaMapper
from ibis.formats.numpy import NumpyType
from ibis.formats.pyarrow import PyArrowData, PyArrowType

_has_arrow_dtype = hasattr(pd, "ArrowDtype")

if not _has_arrow_dtype:
    warnings.warn(
        f"The `ArrowDtype` class is not available in pandas {pd.__version__}. "
        "Install pandas >= 1.5.0 for interop with pandas and arrow dtype support"
    )


class PandasType(NumpyType):
    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if isinstance(typ, pdt.DatetimeTZDtype):
            return dt.Timestamp(timezone=str(typ.tz), nullable=nullable)
        elif pdt.is_datetime64_dtype(typ):
            return dt.Timestamp(nullable=nullable)
        elif isinstance(typ, pdt.CategoricalDtype):
            return dt.String(nullable=nullable)
        elif pdt.is_extension_array_dtype(typ):
            if _has_arrow_dtype and isinstance(typ, pd.ArrowDtype):
                return PyArrowType.to_ibis(typ.pyarrow_dtype, nullable=nullable)
            else:
                name = typ.__class__.__name__.replace("Dtype", "")
                klass = getattr(dt, name)
                return klass(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if dtype.is_timestamp() and dtype.timezone:
            return pdt.DatetimeTZDtype("ns", dtype.timezone)
        elif dtype.is_interval():
            return np.dtype(f"timedelta64[{dtype.unit.short}]")
        else:
            return super().from_ibis(dtype)


class PandasSchema(SchemaMapper):
    @classmethod
    def to_ibis(cls, pandas_schema):
        if isinstance(pandas_schema, pd.Series):
            pandas_schema = pandas_schema.to_list()

        fields = {name: PandasType.to_ibis(t) for name, t in pandas_schema}

        return sch.Schema(fields)

    @classmethod
    def from_ibis(cls, schema):
        names = schema.names
        types = [PandasType.from_ibis(t) for t in schema.types]
        return list(zip(names, types))


class PandasData(DataMapper):
    @classmethod
    def infer_scalar(cls, s):
        return PyArrowData.infer_scalar(s)

    @classmethod
    def infer_column(cls, s):
        return PyArrowData.infer_column(s)

    @classmethod
    def infer_table(cls, df, schema=None):
        schema = schema if schema is not None else {}

        pairs = []
        for column_name in df.dtypes.keys():
            if not isinstance(column_name, str):
                raise TypeError(
                    "Column names must be strings to use the pandas backend"
                )

            if column_name in schema:
                ibis_dtype = schema[column_name]
            else:
                pandas_column = df[column_name]
                pandas_dtype = pandas_column.dtype
                if pandas_dtype == np.object_:
                    ibis_dtype = cls.infer_column(pandas_column)
                else:
                    ibis_dtype = PandasType.to_ibis(pandas_dtype)

            pairs.append((column_name, ibis_dtype))

        return sch.Schema.from_tuples(pairs)

    @classmethod
    def convert_scalar(cls, obj, dtype):
        series = pd.Series([obj])
        casted = cls.convert_column(series, dtype)
        return casted[0]

    @classmethod
    def convert_column(cls, obj, dtype):
        pandas_type = PandasType.from_ibis(dtype)

        method_name = f"convert_{dtype.__class__.__name__}"
        convert_method = getattr(cls, method_name, cls.convert_default)

        result = convert_method(obj, dtype, pandas_type)
        assert not isinstance(result, np.ndarray), f"{convert_method} -> {type(result)}"
        return result

    @classmethod
    def convert_table(cls, df, schema):
        if len(schema) != len(df.columns):
            raise ValueError(
                "schema column count does not match input data column count"
            )

        for name, dtype in schema.items():
            df[name] = cls.convert_column(df[name], dtype)

        return df

    @staticmethod
    def convert_GeoSpatial(s, dtype, pandas_type):
        return s

    convert_Point = (
        convert_LineString
    ) = (
        convert_Polygon
    ) = (
        convert_MultiLineString
    ) = convert_MultiPoint = convert_MultiPolygon = convert_GeoSpatial

    @staticmethod
    def convert_default(s, dtype, pandas_type):
        if pandas_type == object:
            func = lambda x: dt.normalize(dtype, x, none=pd.NA, immutable=False)
            return s.map(func, na_action="ignore")
        else:
            return s.astype(pandas_type)

    @staticmethod
    def convert_Boolean(s, dtype, pandas_type):
        if s.empty:
            return s.astype(pandas_type)
        elif pdt.is_object_dtype(s.dtype):
            return s
        elif s.dtype != pandas_type:
            return s.map(bool, na_action="ignore")
        else:
            return s

    @staticmethod
    def convert_Integer(s, dtype, pandas_type):
        if pdt.is_datetime64_any_dtype(s.dtype):
            return s.astype("int64").floordiv(int(1e9)).astype(pandas_type)
        else:
            return s.astype(pandas_type, errors="ignore")

    convert_SignedInteger = convert_UnsignedInteger = convert_Integer
    convert_Int64 = convert_Int32 = convert_Int16 = convert_Int8 = convert_SignedInteger
    convert_UInt64 = (
        convert_UInt32
    ) = convert_UInt16 = convert_UInt8 = convert_UnsignedInteger

    @staticmethod
    def convert_Floating(s, dtype, pandas_type):
        if pdt.is_datetime64_any_dtype(s.dtype):
            return s.astype("int64").floordiv(int(1e9)).astype(pandas_type)
        else:
            return s.astype(pandas_type, errors="ignore")

    convert_Float64 = convert_Float32 = convert_Float16 = convert_Floating

    @staticmethod
    def convert_Timestamp(s, dtype, pandas_type):
        if isinstance(dtype, pd.DatetimeTZDtype):
            return s.dt.tz_convert(dtype.timezone)
        elif pdt.is_datetime64_dtype(s.dtype):
            return s.dt.tz_localize(dtype.timezone)
        elif pdt.is_numeric_dtype(s.dtype):
            return pd.to_datetime(s, unit="s").dt.tz_localize(dtype.timezone)
        else:
            try:
                return s.astype(pandas_type)
            except pd.errors.OutOfBoundsDatetime:  # uncovered
                try:
                    from dateutil.parser import parse as date_parse

                    return s.map(date_parse, na_action="ignore")
                except TypeError:
                    return s
            except TypeError:
                try:
                    return pd.to_datetime(s).dt.tz_convert(dtype.timezone)
                except TypeError:
                    return pd.to_datetime(s).dt.tz_localize(dtype.timezone)

    @staticmethod
    def convert_Date(s, dtype, pandas_type):
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif pdt.is_numeric_dtype(s.dtype):
            s = pd.to_datetime(s, unit="D")
        else:
            s = pd.to_datetime(s).astype(pandas_type, errors="ignore")

        return s.dt.normalize()

    @staticmethod
    def convert_Interval(s, dtype, pandas_type):
        values = s.values
        try:
            result = values.astype(pandas_type)
        except ValueError:  # can happen when `column` is DateOffsets  # uncovered
            result = s
        else:
            result = s.__class__(result, index=s.index, name=s.name)
        return result

    @staticmethod
    def convert_Decimal(s, dtype, pandas_type):
        import decimal

        import pyarrow as pa

        arr = pa.array(s.map(decimal.Decimal).values).cast(
            dtype.to_pyarrow(), safe=False
        )
        return arr.to_pandas()

    @staticmethod
    def convert_String(s, dtype, pandas_type):
        # TODO(kszucs): should switch to the new pandas string type and convert
        # object columns using s.convert_dtypes() method
        return s.map(str, na_action="ignore").astype(object)

    # @staticmethod
    # def convert_JSON(s, dtype, pandas_type):
    #     # if dtype.is_nested():
    #     #     return s.map(json.dumps, na_action="ignore")
    #     # elif s.dtype != pandas_type:
    #     #     raise TypeError(f"cannot convert {dtype} to JSON")


class DaskData(PandasData):
    @classmethod
    def infer_column(cls, s):
        return PyArrowData.infer_column(s.compute())
