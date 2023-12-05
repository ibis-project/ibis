from __future__ import annotations

import numpy as np
import toolz

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper

_from_numpy_types = toolz.keymap(
    np.dtype,
    {
        np.bool_: dt.Boolean,
        np.int8: dt.Int8,
        np.int16: dt.Int16,
        np.int32: dt.Int32,
        np.int64: dt.Int64,
        np.uint8: dt.UInt8,
        np.uint16: dt.UInt16,
        np.uint32: dt.UInt32,
        np.uint64: dt.UInt64,
        np.float16: dt.Float16,
        np.float32: dt.Float32,
        np.float64: dt.Float64,
    },
)


_to_numpy_types = {v: k for k, v in _from_numpy_types.items()}


class NumpyType(TypeMapper[np.dtype]):
    @classmethod
    def to_ibis(cls, typ: np.dtype, nullable: bool = True) -> dt.DataType:
        if np.issubdtype(typ, np.datetime64):
            # TODO(kszucs): the following code provedes proper timestamp roundtrips
            # between ibis and numpy/pandas but breaks the test suite at several
            # places, we should revisit this later
            # unit, _ = np.datetime_data(typ)
            # if unit in {'generic', 'Y', 'M', 'D', 'h', 'm'}:
            #     return dt.Timestamp(nullable=nullable)
            # else:
            #     return dt.Timestamp.from_unit(unit, nullable=nullable)
            return dt.Timestamp(nullable=nullable)
        elif np.issubdtype(typ, np.timedelta64):
            unit, _ = np.datetime_data(typ)
            if unit == "generic":
                unit = "s"
            return dt.Interval(unit, nullable=nullable)
        elif np.issubdtype(typ, np.str_):
            return dt.String(nullable=nullable)
        elif np.issubdtype(typ, np.bytes_):
            return dt.Binary(nullable=nullable)
        else:
            try:
                return _from_numpy_types[typ](nullable=nullable)
            except KeyError:
                raise TypeError(f"numpy dtype {typ!r} is not supported")

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> np.dtype:
        if dtype.is_interval():
            return np.dtype(f"timedelta64[{dtype.unit.short}]")
        elif dtype.is_timestamp():
            # TODO(kszucs): the following code provedes proper timestamp roundtrips
            # between ibis and numpy/pandas but breaks the test suite at several
            # places, we should revisit this later
            # return np.dtype(f"datetime64[{dtype.unit.short}]")
            return np.dtype("datetime64[ns]")
        elif dtype.is_date():
            # return np.dtype("datetime64[D]")
            return np.dtype("datetime64[ns]")
        elif dtype.is_time():
            return np.dtype("timedelta64[ns]")
        elif (
            dtype.is_null()
            or dtype.is_decimal()
            or dtype.is_struct()
            or dtype.is_variadic()
            or dtype.is_unknown()
            or dtype.is_uuid()
            or dtype.is_geospatial()
            or dtype.is_inet()
            or dtype.is_macaddr()
        ):
            return np.dtype("object")
        else:
            try:
                return _to_numpy_types[type(dtype)]
            except KeyError:
                raise TypeError(f"ibis dtype {dtype!r} is not supported")


class NumpySchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema):
        numpy_types = map(NumpyType.from_ibis, schema.types)
        return list(zip(schema.names, numpy_types))

    @classmethod
    def to_ibis(cls, schema):
        ibis_types = {name: NumpyType.to_ibis(typ) for name, typ in schema}
        return sch.Schema(ibis_types)
