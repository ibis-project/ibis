from __future__ import annotations

from typing import Literal

import sqlglot.expressions as sge

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.base.sqlglot.datatypes import SqlglotType
from ibis.common.collections import FrozenDict

typecode = sge.DataType.Type


# TODO(kszucs): add a bool converter method to support different clickhouse bool types
def _bool_type() -> Literal["Bool", "UInt8", "Int8"]:
    return getattr(getattr(ibis.options, "clickhouse", None), "bool_type", "Bool")


class ClickhouseType(SqlglotType):
    dialect = "clickhouse"
    default_decimal_precision = None
    default_decimal_scale = None
    default_nullable = False

    unknown_type_strings = FrozenDict(
        {
            "ipv4": dt.INET(nullable=default_nullable),
            "ipv6": dt.INET(nullable=default_nullable),
            "object('json')": dt.JSON(nullable=default_nullable),
            "array(null)": dt.Array(dt.null, nullable=default_nullable),
            "array(nothing)": dt.Array(dt.null, nullable=default_nullable),
        }
    )

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sge.DataType:
        """Convert a sqlglot type to an ibis type."""
        typ = super().from_ibis(dtype)
        if dtype.nullable and not dtype.is_map():
            # map cannot be nullable in clickhouse
            return sge.DataType(this=typecode.NULLABLE, expressions=[typ])
        else:
            return typ

    @classmethod
    def _from_sqlglot_NULLABLE(cls, inner_type: sge.DataType) -> dt.DataType:
        return cls.to_ibis(inner_type, nullable=True)

    @classmethod
    def _from_sqlglot_DATETIME(
        cls, timezone: sge.DataTypeParam | None = None
    ) -> dt.Timestamp:
        return dt.Timestamp(
            scale=0,
            timezone=None if timezone is None else timezone.this.this,
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_DATETIME64(
        cls,
        scale: sge.DataTypeSize | None = None,
        timezone: sge.Literal | None = None,
    ) -> dt.Timestamp:
        return dt.Timestamp(
            timezone=None if timezone is None else timezone.this.this,
            scale=int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_LOWCARDINALITY(cls, inner_type: sge.DataType) -> dt.DataType:
        return cls.to_ibis(inner_type)

    @classmethod
    def _from_sqlglot_NESTED(cls, *fields: sge.DataType) -> dt.Struct:
        fields = {
            field.name: dt.Array(
                cls.to_ibis(field.args["kind"]), nullable=cls.default_nullable
            )
            for field in fields
        }
        return dt.Struct(fields, nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_Timestamp(cls, dtype: dt.Timestamp) -> sge.DataType:
        if dtype.timezone is None:
            timezone = None
        else:
            timezone = sge.DataTypeParam(this=sge.Literal.string(dtype.timezone))

        if dtype.scale is None:
            return sge.DataType(this=typecode.DATETIME, expressions=[timezone])
        else:
            scale = sge.DataTypeParam(this=sge.Literal.number(dtype.scale))
            return sge.DataType(this=typecode.DATETIME64, expressions=[scale, timezone])

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        # key cannot be nullable in clickhouse
        key_type = cls.from_ibis(dtype.key_type.copy(nullable=False))
        value_type = cls.from_ibis(dtype.value_type)
        return sge.DataType(this=typecode.MAP, expressions=[key_type, value_type])
