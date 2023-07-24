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
        if pdt.is_datetime64tz_dtype(typ):
            return dt.Timestamp(timezone=str(typ.tz), nullable=nullable)
        elif pdt.is_datetime64_dtype(typ):
            return dt.Timestamp(nullable=nullable)
        elif pdt.is_categorical_dtype(typ):
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
            return pdt.DatetimeTZDtype('ns', dtype.timezone)
        elif dtype.is_interval():
            return np.dtype(f'timedelta64[{dtype.unit.short}]')
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
                    'Column names must be strings to use the pandas backend'
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
    def convert_table(cls, df, schema):
        if len(schema) != len(df.columns):
            raise ValueError(
                "schema column count does not match input data column count"
            )

        for (name, series), dtype in zip(df.items(), schema.types):
            df[name] = cls.convert_column(series, dtype)

        # return data with the schema's columns which may be different than the
        # input columns
        df.columns = schema.names
        return df

    @classmethod
    def convert_column(cls, obj, dtype):
        pandas_type = PandasType.from_ibis(dtype)

        if obj.dtype == pandas_type and dtype.is_primitive():
            return obj

        method_name = f"convert_{dtype.__class__.__name__}"
        convert_method = getattr(cls, method_name, cls.convert_default)

        result = convert_method(obj, dtype, pandas_type)
        assert not isinstance(result, np.ndarray), f"{convert_method} -> {type(result)}"
        return result

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
        try:
            return s.astype(pandas_type)
        except Exception:  # noqa: BLE001
            return s

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
    def convert_Timestamp(s, dtype, pandas_type):
        import pandas.api.types as pdt

        if pdt.is_datetime64tz_dtype(s.dtype):
            return s.dt.tz_convert(dtype.timezone)
        elif pdt.is_datetime64_dtype(s.dtype):
            return s.dt.tz_localize(dtype.timezone)
        else:
            import pandas as pd

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
        import pandas.api.types as pdt

        if pdt.is_datetime64tz_dtype(s.dtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.astype(pandas_type, errors='ignore').dt.normalize()

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
    def convert_String(s, dtype, pandas_type):
        return s.astype(pandas_type, errors='ignore')

    @staticmethod
    def convert_UUID(s, dtype, pandas_type):
        from uuid import UUID

        return s.map(
            lambda v: v if isinstance(v, UUID) else UUID(v), na_action="ignore"
        )

    @staticmethod
    def convert_Struct(s, dtype, pandas_type):
        return s.map(
            lambda values, names=dtype.names: (
                values if isinstance(values, dict) else dict(zip(names, values))
            ),
            na_action="ignore",
        )

    @staticmethod
    def convert_Array(s, dtype, pandas_type):
        return s.map(list, na_action="ignore")

    @staticmethod
    def convert_Map(s, dtype, pandas_type):
        return s.map(dict, na_action="ignore")

    @staticmethod
    def convert_JSON(s, dtype, pandas_type):
        import json

        def try_json(x):
            try:
                return json.loads(x)
            except (TypeError, json.JSONDecodeError):
                return x

        return s.map(try_json, na_action="ignore").astype("object")


class DaskData(PandasData):
    @classmethod
    def infer_column(cls, s):
        return PyArrowData.infer_column(s.compute())
