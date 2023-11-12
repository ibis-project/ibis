from __future__ import annotations

import google.cloud.bigquery as bq
import sqlglot.expressions as sge

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sqlglot.datatypes import SqlglotType
from ibis.formats import SchemaMapper


class BigQueryType(SqlglotType):
    dialect = "bigquery"

    default_decimal_precision = 38
    default_decimal_scale = 9

    @classmethod
    def _from_sqlglot_NUMERIC(cls) -> dt.Decimal:
        return dt.Decimal(
            cls.default_decimal_precision,
            cls.default_decimal_scale,
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_BIGNUMERIC(cls) -> dt.Decimal:
        return dt.Decimal(76, 38, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_DATETIME(cls) -> dt.Decimal:
        return dt.Timestamp(timezone=None, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP(cls) -> dt.Decimal:
        return dt.Timestamp(timezone="UTC", nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_GEOGRAPHY(cls) -> dt.Decimal:
        return dt.GeoSpatial(
            geotype="geography", srid=4326, nullable=cls.default_nullable
        )

    @classmethod
    def _from_sqlglot_TINYINT(cls) -> dt.Int64:
        return dt.Int64(nullable=cls.default_nullable)

    _from_sqlglot_UINT = (
        _from_sqlglot_USMALLINT
    ) = (
        _from_sqlglot_UTINYINT
    ) = _from_sqlglot_INT = _from_sqlglot_SMALLINT = _from_sqlglot_TINYINT

    @classmethod
    def _from_sqlglot_UBIGINT(cls) -> dt.Int64:
        raise TypeError("Unsigned BIGINT isn't representable in BigQuery INT64")

    @classmethod
    def _from_sqlglot_FLOAT(cls) -> dt.Double:
        return dt.Float64(nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_MAP(cls) -> dt.Map:
        raise NotImplementedError(
            "Cannot convert sqlglot Map type to ibis type: maps are not supported in BigQuery"
        )

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        raise NotImplementedError(
            "Cannot convert Ibis Map type to BigQuery type: maps are not supported in BigQuery"
        )

    @classmethod
    def _from_ibis_Timestamp(cls, dtype: dt.Timestamp) -> sge.DataType:
        if dtype.timezone is None:
            return sge.DataType(this=sge.DataType.Type.DATETIME)
        elif dtype.timezone == "UTC":
            return sge.DataType(this=sge.DataType.Type.TIMESTAMPTZ)
        else:
            raise TypeError(
                "BigQuery does not support timestamps with timezones other than 'UTC'"
            )

    @classmethod
    def _from_ibis_Decimal(cls, dtype: dt.Decimal) -> sge.DataType:
        precision = dtype.precision
        scale = dtype.scale
        if (precision, scale) == (76, 38):
            return sge.DataType(this=sge.DataType.Type.BIGDECIMAL)
        elif (precision, scale) in ((38, 9), (None, None)):
            return sge.DataType(this=sge.DataType.Type.DECIMAL)
        else:
            raise TypeError(
                "BigQuery only supports decimal types with precision of 38 and "
                f"scale of 9 (NUMERIC) or precision of 76 and scale of 38 (BIGNUMERIC). "
                f"Current precision: {dtype.precision}. Current scale: {dtype.scale}"
            )

    @classmethod
    def _from_ibis_UInt64(cls, dtype: dt.UInt64) -> sge.DataType:
        raise TypeError(
            f"Conversion from {dtype} to BigQuery integer type (Int64) is lossy"
        )

    @classmethod
    def _from_ibis_UInt32(cls, dtype: dt.UInt32) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.BIGINT)

    _from_ibis_UInt8 = _from_ibis_UInt16 = _from_ibis_UInt32

    @classmethod
    def _from_ibis_GeoSpatial(cls, dtype: dt.GeoSpatial) -> sge.DataType:
        if (dtype.geotype, dtype.srid) == ("geography", 4326):
            return sge.DataType(this=sge.DataType.Type.GEOGRAPHY)
        else:
            raise TypeError(
                "BigQuery geography uses points on WGS84 reference ellipsoid."
                f"Current geotype: {dtype.geotype}, Current srid: {dtype.srid}"
            )


class BigQuerySchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema) -> list[bq.SchemaField]:
        schema_fields = []

        for name, typ in ibis.schema(schema).items():
            if typ.is_array():
                value_type = typ.value_type
                if value_type.is_array():
                    raise TypeError("Nested arrays are not supported in BigQuery")

                is_struct = value_type.is_struct()

                field_type = (
                    "RECORD" if is_struct else BigQueryType.to_string(typ.value_type)
                )
                mode = "REPEATED"
                fields = cls.from_ibis(ibis.schema(getattr(value_type, "fields", {})))
            elif typ.is_struct():
                field_type = "RECORD"
                mode = "NULLABLE" if typ.nullable else "REQUIRED"
                fields = cls.from_ibis(ibis.schema(typ.fields))
            else:
                field_type = BigQueryType.to_string(typ)
                mode = "NULLABLE" if typ.nullable else "REQUIRED"
                fields = ()

            schema_fields.append(
                bq.SchemaField(name, field_type=field_type, mode=mode, fields=fields)
            )
        return schema_fields

    @classmethod
    def _dtype_from_bigquery_field(cls, field: bq.SchemaField) -> dt.DataType:
        typ = field.field_type
        if typ == "RECORD":
            assert field.fields, "RECORD fields are empty"
            fields = {f.name: cls._dtype_from_bigquery_field(f) for f in field.fields}
            dtype = dt.Struct(fields)
        else:
            dtype = BigQueryType.from_string(typ)

        mode = field.mode
        if mode == "NULLABLE":
            return dtype.copy(nullable=True)
        elif mode == "REQUIRED":
            return dtype.copy(nullable=False)
        elif mode == "REPEATED":
            # arrays with NULL elements aren't supported
            return dt.Array(dtype.copy(nullable=False))
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
        raise NotImplementedError("Maps are not supported in BigQuery")
    yield dt
