from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.expr.schema import Schema
from ibis.formats import DataMapper, SchemaMapper, TableProxy, TypeMapper
from ibis.util import V

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import polars as pl
    import pyarrow.dataset as ds


_from_pyarrow_types = {
    pa.int8(): dt.Int8,
    pa.int16(): dt.Int16,
    pa.int32(): dt.Int32,
    pa.int64(): dt.Int64,
    pa.uint8(): dt.UInt8,
    pa.uint16(): dt.UInt16,
    pa.uint32(): dt.UInt32,
    pa.uint64(): dt.UInt64,
    pa.float16(): dt.Float16,
    pa.float32(): dt.Float32,
    pa.float64(): dt.Float64,
    pa.string(): dt.String,
    pa.binary(): dt.Binary,
    pa.bool_(): dt.Boolean,
    pa.date32(): dt.Date,
    pa.date64(): dt.Date,
    pa.null(): dt.Null,
    pa.string(): dt.String,
    pa.large_binary(): dt.Binary,
    pa.large_string(): dt.String,
    pa.binary(): dt.Binary,
}

with contextlib.suppress(AttributeError):
    _from_pyarrow_types[pa.string_view()] = dt.String


_to_pyarrow_types = {
    dt.Null: pa.null(),
    dt.Boolean: pa.bool_(),
    dt.Binary: pa.binary(),
    dt.Int8: pa.int8(),
    dt.Int16: pa.int16(),
    dt.Int32: pa.int32(),
    dt.Int64: pa.int64(),
    dt.UInt8: pa.uint8(),
    dt.UInt16: pa.uint16(),
    dt.UInt32: pa.uint32(),
    dt.UInt64: pa.uint64(),
    dt.Float16: pa.float16(),
    dt.Float32: pa.float32(),
    dt.Float64: pa.float64(),
    dt.String: pa.string(),
    dt.Binary: pa.binary(),
    # assume unknown types can be converted into strings
    dt.Unknown: pa.string(),
    dt.MACADDR: pa.string(),
    dt.INET: pa.string(),
    dt.UUID: pa.string(),
    dt.JSON: pa.string(),
}


class PyArrowType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: pa.DataType, nullable=True) -> dt.DataType:
        """Convert a pyarrow type to an ibis type."""
        if pa.types.is_null(typ):
            return dt.null
        elif pa.types.is_decimal(typ):
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif pa.types.is_timestamp(typ):
            return dt.Timestamp.from_unit(typ.unit, timezone=typ.tz, nullable=nullable)
        elif pa.types.is_time(typ):
            return dt.Time(nullable=nullable)
        elif pa.types.is_duration(typ):
            return dt.Interval(typ.unit, nullable=nullable)
        elif pa.types.is_interval(typ):
            raise ValueError("Arrow interval type is not supported")
        elif pa.types.is_fixed_size_list(typ):
            value_dtype = cls.to_ibis(typ.value_type, typ.value_field.nullable)
            return dt.Array(value_dtype, length=typ.list_size, nullable=nullable)
        elif pa.types.is_list(typ) or pa.types.is_large_list(typ):
            value_dtype = cls.to_ibis(typ.value_type, typ.value_field.nullable)
            return dt.Array(value_dtype, nullable=nullable)
        elif pa.types.is_struct(typ):
            field_dtypes = {
                field.name: cls.to_ibis(field.type, field.nullable) for field in typ
            }
            return dt.Struct(field_dtypes, nullable=nullable)
        elif pa.types.is_map(typ):
            # TODO(kszucs): keys_sorted has just been exposed in pyarrow
            key_dtype = cls.to_ibis(typ.key_type, typ.key_field.nullable)
            value_dtype = cls.to_ibis(typ.item_type, typ.item_field.nullable)
            return dt.Map(key_dtype, value_dtype, nullable=nullable)
        elif pa.types.is_dictionary(typ):
            return cls.to_ibis(typ.value_type)
        elif (
            isinstance(typ, pa.ExtensionType)
            and type(typ).__module__ == "geoarrow.types.type_pyarrow"
        ):
            from geoarrow import types as gat

            gat.type_pyarrow.register_extension_types()

            auth_code = None
            if typ.crs is not None:
                crs_dict = typ.crs.to_json_dict()
                if "id" in crs_dict:
                    crs_id = crs_dict["id"]
                    if "authority" in crs_id and "code" in crs_id:
                        auth_code = (crs_id["authority"], crs_id["code"])

            if typ.crs is not None and auth_code is None:
                # It is possible to have PROJJSON that does not have an authority/code
                # attached, either because the producer didn't have that information
                # (e.g., because they were reading a older shapefile). In this case,
                # pyproj can often guess the authority/code.
                import pyproj

                auth_code = pyproj.CRS(typ.crs.to_json()).to_authority()
                if auth_code is None:
                    raise ValueError(f"Can't resolve SRID of crs {typ.crs}")

            if auth_code is None:
                srid = None
            elif auth_code == ("OGC", "CRS84"):
                # OGC:CRS84 and EPSG:4326 are identical except for the order of
                # coordinates (i.e., lon lat vs. lat lon) in their official definition.
                # This axis ordering is ignored in all but the most obscure scenarios
                # such that these are identical. OGC:CRS84 is more correct, but EPSG:4326
                # is more common.
                srid = 4326
            else:
                # This works because the two most common srid authorities are EPSG and ESRI
                # and the "codes" are all integers and don't intersect with each other on
                # purpose. This won't scale to something like OGC:CRS27 (not common).
                srid = int(auth_code[1])

            if typ.edge_type == gat.EdgeType.SPHERICAL:
                geotype = "geography"
            else:
                geotype = "geometry"

            return dt.GeoSpatial(geotype, srid, nullable)
        else:
            return _from_pyarrow_types[typ](nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> pa.DataType:
        """Convert an ibis type to a pyarrow type."""
        if dtype.is_decimal():
            # set default precision and scale to something; unclear how to choose this
            precision = 38 if dtype.precision is None else dtype.precision
            scale = 9 if dtype.scale is None else dtype.scale

            if precision > 76:
                raise ValueError(
                    f"Unsupported precision {dtype.precision} for decimal type"
                )
            elif precision > 38:
                return pa.decimal256(precision, scale)
            else:
                return pa.decimal128(precision, scale)
        elif dtype.is_timestamp():
            return pa.timestamp(
                dtype.unit.short if dtype.scale is not None else "us", tz=dtype.timezone
            )
        elif dtype.is_interval():
            short = dtype.unit.short
            if short in {"ns", "us", "ms", "s"}:
                return pa.duration(short)
            else:
                return pa.month_day_nano_interval()
        elif dtype.is_time():
            return pa.time64("ns")
        elif dtype.is_date():
            return pa.date32()
        elif dtype.is_array():
            value_field = pa.field(
                "item",
                cls.from_ibis(dtype.value_type),
                nullable=dtype.value_type.nullable,
            )
            if dtype.length is None:
                return pa.list_(value_field)
            else:
                return pa.list_(value_field, dtype.length)
        elif dtype.is_struct():
            fields = [
                pa.field(name, cls.from_ibis(dtype), nullable=dtype.nullable)
                for name, dtype in dtype.items()
            ]
            return pa.struct(fields)
        elif dtype.is_map():
            key_field = pa.field(
                "key",
                cls.from_ibis(dtype.key_type),
                nullable=False,  # pyarrow doesn't allow nullable keys
            )
            value_field = pa.field(
                "value",
                cls.from_ibis(dtype.value_type),
                nullable=dtype.value_type.nullable,
            )
            return pa.map_(key_field, value_field, keys_sorted=False)
        elif dtype.is_geospatial():
            from geoarrow import types as gat

            # Resolve CRS
            if dtype.srid is None:
                crs = None
            elif dtype.srid == 4326:
                crs = gat.OGC_CRS84
            else:
                import pyproj

                # Assume that these are EPSG codes. An srid is more accurately a key
                # into a backend/connection-specific lookup table; however, most usage
                # should work with this assumption.
                crs = pyproj.CRS(f"EPSG:{dtype.srid}")

            # Resolve edge type
            if dtype.geotype == "geography":
                edge_type = gat.EdgeType.SPHERICAL
            else:
                edge_type = gat.EdgeType.PLANAR

            return gat.wkb(crs=crs, edge_type=edge_type).to_pyarrow()
        else:
            try:
                return _to_pyarrow_types[type(dtype)]
            except KeyError:
                raise NotImplementedError(
                    f"Converting {dtype} to pyarrow is not supported yet"
                )


class PyArrowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: Schema) -> pa.Schema:
        """Convert a schema to a pyarrow schema."""
        fields = [
            pa.field(name, PyArrowType.from_ibis(dtype), nullable=dtype.nullable)
            for name, dtype in schema.items()
        ]
        return pa.schema(fields)

    @classmethod
    def to_ibis(cls, schema: pa.Schema) -> Schema:
        """Convert a pyarrow schema to a schema."""
        fields = [(f.name, PyArrowType.to_ibis(f.type, f.nullable)) for f in schema]
        return Schema.from_tuples(fields)


class PyArrowData(DataMapper):
    @classmethod
    def infer_scalar(cls, scalar: Any) -> dt.DataType:
        """Infer the ibis type of a scalar."""
        return PyArrowType.to_ibis(pa.scalar(scalar).type)

    @classmethod
    def infer_column(cls, column: Sequence) -> dt.DataType:
        """Infer the ibis type of a sequence."""
        if isinstance(column, pa.Array):
            return PyArrowType.to_ibis(column.type)

        try:
            pyarrow_type = pa.array(column, from_pandas=True).type
            # pyarrow_type = pa.infer_type(column, from_pandas=True)
        except pa.ArrowInvalid:
            try:
                # handle embedded series objects
                return dt.highest_precedence(map(dt.infer, column))
            except TypeError:
                # we can still have a type error, e.g., float64 and string in the
                # same array
                return dt.unknown
        except pa.ArrowTypeError:
            # arrow can't infer the type
            return dt.unknown
        else:
            # arrow inferred the type, now convert that type to an ibis type
            return PyArrowType.to_ibis(pyarrow_type)

    @classmethod
    def infer_table(cls, table) -> Schema:
        """Infer the schema of a table."""
        if not isinstance(table, pa.Table):
            table = pa.table(table)

        return PyArrowSchema.to_ibis(table.schema)

    @classmethod
    def convert_scalar(cls, scalar: pa.Scalar, dtype: dt.DataType) -> pa.Scalar:
        desired_type = PyArrowType.from_ibis(dtype)
        scalar_type = scalar.type
        if scalar_type != desired_type:
            try:
                return scalar.cast(desired_type)
            except pa.ArrowNotImplementedError:
                # pyarrow doesn't support some scalar casts that are supported
                # when using arrays or tables
                return pa.scalar(scalar.as_py(), type=scalar_type).cast(desired_type)
        else:
            return scalar

    @classmethod
    def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
        desired_type = PyArrowType.from_ibis(dtype)
        if column.type != desired_type:
            return column.cast(desired_type)
        else:
            return column

    @classmethod
    def convert_table(cls, table: pa.Table, schema: Schema) -> pa.Table:
        desired_schema = PyArrowSchema.from_ibis(schema)
        if table.schema == desired_schema:
            return table
        arrays = [
            cls.convert_column(table[name], dtype) for name, dtype in schema.items()
        ]
        return pa.Table.from_arrays(arrays, schema=desired_schema)


class PyArrowTableProxy(TableProxy[V]):
    def to_frame(self):
        return self.obj.to_pandas()

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        return self.obj

    def to_polars(self, schema: Schema) -> pl.DataFrame:
        import polars as pl

        from ibis.formats.polars import PolarsData

        df = pl.from_arrow(self.obj)
        return PolarsData.convert_table(df, schema)


class PyArrowDatasetProxy(TableProxy[V]):
    ERROR_MESSAGE = """\
You are trying to use a PyArrow Dataset with a backend that will require
materializing the entire dataset in local memory.

If you would like to materialize this dataset, please construct the memtable
directly by running `ibis.memtable(my_dataset.to_table())`."""

    __slots__ = ("obj",)
    obj: V

    def __init__(self, obj: V) -> None:
        self.obj = obj

    # pyarrow datasets are hashable, so we override the hash from TableProxy
    def __hash__(self):
        return hash(self.obj)

    def to_frame(self) -> pd.DataFrame:
        raise com.UnsupportedOperationError(self.ERROR_MESSAGE)

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        raise com.UnsupportedOperationError(self.ERROR_MESSAGE)

    def to_pyarrow_dataset(self, schema: Schema) -> ds.Dataset:
        """Return the dataset object itself.

        Use with backends that can perform pushdowns into dataset objects.
        """
        return self.obj

    def to_polars(self, schema: Schema) -> pa.Table:
        raise com.UnsupportedOperationError(self.ERROR_MESSAGE)
