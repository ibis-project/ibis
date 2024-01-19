from __future__ import annotations

import google.cloud.bigquery as bq

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sqlglot.datatypes import BigQueryType
from ibis.formats import SchemaMapper


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
