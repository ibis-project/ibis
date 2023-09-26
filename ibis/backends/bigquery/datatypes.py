from __future__ import annotations

import google.cloud.bigquery as bq
import sqlglot as sg

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper

_from_bigquery_types = {
    "INT64": dt.Int64,
    "INTEGER": dt.Int64,
    "FLOAT": dt.Float64,
    "FLOAT64": dt.Float64,
    "BOOL": dt.Boolean,
    "BOOLEAN": dt.Boolean,
    "STRING": dt.String,
    "DATE": dt.Date,
    "TIME": dt.Time,
    "BYTES": dt.Binary,
    "JSON": dt.JSON,
}


class BigQueryType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: str, nullable: bool = True) -> dt.DataType:
        if typ == "DATETIME":
            return dt.Timestamp(timezone=None, nullable=nullable)
        elif typ == "TIMESTAMP":
            return dt.Timestamp(timezone="UTC", nullable=nullable)
        elif typ == "NUMERIC":
            return dt.Decimal(38, 9, nullable=nullable)
        elif typ == "BIGNUMERIC":
            return dt.Decimal(76, 38, nullable=nullable)
        elif typ == "GEOGRAPHY":
            return dt.GeoSpatial(geotype="geography", srid=4326, nullable=nullable)
        else:
            try:
                return _from_bigquery_types[typ](nullable=nullable)
            except KeyError:
                raise TypeError(f"Unable to convert BigQuery type to ibis: {typ}")

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> str:
        if dtype.is_floating():
            return "FLOAT64"
        elif dtype.is_uint64():
            raise TypeError(
                "Conversion from uint64 to BigQuery integer type (int64) is lossy"
            )
        elif dtype.is_integer():
            return "INT64"
        elif dtype.is_binary():
            return "BYTES"
        elif dtype.is_date():
            return "DATE"
        elif dtype.is_timestamp():
            if dtype.timezone is None:
                return "DATETIME"
            elif dtype.timezone == "UTC":
                return "TIMESTAMP"
            else:
                raise TypeError(
                    "BigQuery does not support timestamps with timezones other than 'UTC'"
                )
        elif dtype.is_decimal():
            if (dtype.precision, dtype.scale) == (76, 38):
                return "BIGNUMERIC"
            if (dtype.precision, dtype.scale) in [(38, 9), (None, None)]:
                return "NUMERIC"
            raise TypeError(
                "BigQuery only supports decimal types with precision of 38 and "
                f"scale of 9 (NUMERIC) or precision of 76 and scale of 38 (BIGNUMERIC). "
                f"Current precision: {dtype.precision}. Current scale: {dtype.scale}"
            )
        elif dtype.is_array():
            return f"ARRAY<{cls.from_ibis(dtype.value_type)}>"
        elif dtype.is_struct():
            fields = (
                f"{sg.to_identifier(k).sql('bigquery')} {cls.from_ibis(v)}"
                for k, v in dtype.fields.items()
            )
            return "STRUCT<{}>".format(", ".join(fields))
        elif dtype.is_json():
            return "JSON"
        elif dtype.is_geospatial():
            if (dtype.geotype, dtype.srid) == ("geography", 4326):
                return "GEOGRAPHY"
            raise TypeError(
                "BigQuery geography uses points on WGS84 reference ellipsoid."
                f"Current geotype: {dtype.geotype}, Current srid: {dtype.srid}"
            )
        else:
            return str(dtype).upper()


class BigQuerySchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema) -> list[bq.SchemaField]:
        result = []
        for name, dtype in schema.items():
            if isinstance(dtype, dt.Array):
                mode = "REPEATED"
                dtype = dtype.value_type
            else:
                mode = "REQUIRED" if not dtype.nullable else "NULLABLE"
            field = bq.SchemaField(name, BigQueryType.from_ibis(dtype), mode=mode)
            result.append(field)
        return result

    @classmethod
    def _dtype_from_bigquery_field(cls, field: bq.SchemaField) -> dt.DataType:
        typ = field.field_type
        if typ == "RECORD":
            assert field.fields, "RECORD fields are empty"
            fields = {f.name: cls._dtype_from_bigquery_field(f) for f in field.fields}
            dtype = dt.Struct(fields)
        else:
            dtype = BigQueryType.to_ibis(typ)

        mode = field.mode
        if mode == "NULLABLE":
            return dtype.copy(nullable=True)
        elif mode == "REQUIRED":
            return dtype.copy(nullable=False)
        elif mode == "REPEATED":
            return dt.Array(dtype)
        else:
            raise TypeError(f"Unknown BigQuery field.mode: {mode}")

    @classmethod
    def to_ibis(cls, fields: list[bq.SchemaField]) -> sch.Schema:
        return sch.Schema({f.name: cls._dtype_from_bigquery_field(f) for f in fields})


# TODO(kszucs): we can eliminate this function by making dt.DataType traversible
# using ibis.common.graph.Node, similarly to how we traverse ops.Node instances:
# node.find(types)
def spread_type(dt: dt.DataType):
    """Returns a generator that contains all the types in the given type.

    For complex types like set and array, it returns the types of the elements.
    """
    if dt.is_array():
        yield from spread_type(dt.value_type)
    elif dt.is_struct():
        for type_ in dt.types:
            yield from spread_type(type_)
    elif dt.is_map():
        yield from spread_type(dt.key_type)
        yield from spread_type(dt.value_type)
    yield dt
