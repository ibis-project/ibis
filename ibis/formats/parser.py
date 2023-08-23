from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import sqlglot as sg
from sqlglot.expressions import DataType

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
from ibis.common.collections import FrozenDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlglot.expressions import ColumnDef

    try:
        from sqlglot.expressions import DataTypeParam
    except ImportError:
        from sqlglot.expressions import DataTypeSize as DataTypeParam

SQLGLOT_TYPE_TO_IBIS_TYPE = {
    DataType.Type.BIGDECIMAL: dt.Decimal(76, 38),
    DataType.Type.BIGINT: dt.int64,
    DataType.Type.BINARY: dt.binary,
    DataType.Type.BIT: dt.string,
    DataType.Type.BOOLEAN: dt.boolean,
    DataType.Type.CHAR: dt.string,
    DataType.Type.DATE: dt.date,
    DataType.Type.DOUBLE: dt.float64,
    DataType.Type.ENUM: dt.string,
    DataType.Type.ENUM8: dt.string,
    DataType.Type.ENUM16: dt.string,
    DataType.Type.FLOAT: dt.float32,
    DataType.Type.FIXEDSTRING: dt.string,
    DataType.Type.GEOMETRY: dt.geometry,
    DataType.Type.GEOGRAPHY: dt.geography,
    DataType.Type.HSTORE: dt.Map(dt.string, dt.string),
    DataType.Type.INET: dt.inet,
    DataType.Type.INT128: dt.Decimal(38, 0),
    DataType.Type.INT256: dt.Decimal(76, 0),
    DataType.Type.INT: dt.int32,
    DataType.Type.IPADDRESS: dt.inet,
    DataType.Type.JSON: dt.json,
    DataType.Type.JSONB: dt.json,
    DataType.Type.LONGBLOB: dt.binary,
    DataType.Type.LONGTEXT: dt.string,
    DataType.Type.MEDIUMBLOB: dt.binary,
    DataType.Type.MEDIUMTEXT: dt.string,
    DataType.Type.MONEY: dt.int64,
    DataType.Type.NCHAR: dt.string,
    DataType.Type.NULL: dt.null,
    DataType.Type.NVARCHAR: dt.string,
    DataType.Type.OBJECT: dt.Map(dt.string, dt.json),
    DataType.Type.SMALLINT: dt.int16,
    DataType.Type.SMALLMONEY: dt.int32,
    DataType.Type.TEXT: dt.string,
    DataType.Type.TIME: dt.time,
    DataType.Type.TIMETZ: dt.time,
    DataType.Type.TINYINT: dt.int8,
    DataType.Type.UBIGINT: dt.uint64,
    DataType.Type.UINT: dt.uint32,
    DataType.Type.USMALLINT: dt.uint16,
    DataType.Type.UTINYINT: dt.uint8,
    DataType.Type.UUID: dt.uuid,
    DataType.Type.VARBINARY: dt.binary,
    DataType.Type.VARCHAR: dt.string,
    DataType.Type.VARIANT: dt.json,
    DataType.Type.UNIQUEIDENTIFIER: dt.uuid,
    #############################
    # Unsupported sqlglot types #
    #############################
    # BIGSERIAL = auto()
    # DATETIME64 = auto() # clickhouse
    # ENUM = auto()
    # INT4RANGE = auto()
    # INT4MULTIRANGE = auto()
    # INT8RANGE = auto()
    # INT8MULTIRANGE = auto()
    # NUMRANGE = auto()
    # NUMMULTIRANGE = auto()
    # TSRANGE = auto()
    # TSMULTIRANGE = auto()
    # TSTZRANGE = auto()
    # TSTZMULTIRANGE = auto()
    # DATERANGE = auto()
    # DATEMULTIRANGE = auto()
    # HLLSKETCH = auto()
    # IMAGE = auto()
    # IPPREFIX = auto()
    # ROWVERSION = auto()
    # SERIAL = auto()
    # SET = auto()
    # SMALLSERIAL = auto()
    # SUPER = auto()
    # TIMESTAMPLTZ = auto()
    # UNKNOWN = auto()  # Sentinel value, useful for type annotation
    # UINT128 = auto()
    # UINT256 = auto()
    # USERDEFINED = "USER-DEFINED"
    # XML = auto()
}


class TypeParser(abc.ABC):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def dialect(self) -> str:
        """The dialect this parser is for."""

    default_nullable = True
    """Default nullability when not specified."""

    default_decimal_precision: int | None = None
    """Default decimal precision when not specified."""

    default_decimal_scale: int | None = None
    """Default decimal scale when not specified."""

    default_temporal_scale: int | None = None
    """Default temporal scale when not specified."""

    default_interval_precision: str | None = None
    """Default interval precision when not specified."""

    short_circuit: Mapping[str, dt.DataType] = FrozenDict()
    """Default short-circuit mapping of SQL string types to ibis types."""

    @classmethod
    def parse(cls, text: str) -> dt.DataType:
        """Parse a type string into an ibis data type."""
        short_circuit = cls.short_circuit
        if dtype := short_circuit.get(text, short_circuit.get(text.upper())):
            return dtype
        return cls._get_type(sg.parse_one(text, into=DataType, read=cls.dialect))

    @classmethod
    def _get_ARRAY(cls, value_type: DataType) -> dt.Array:
        return dt.Array(cls._get_type(value_type), nullable=cls.default_nullable)

    @classmethod
    def _get_MAP(cls, key_type: DataType, value_type: DataType) -> dt.Map:
        return dt.Map(
            cls._get_type(key_type),
            cls._get_type(value_type),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_STRUCT(cls, *fields: ColumnDef) -> dt.Struct:
        return dt.Struct(
            {field.name: cls._get_type(field.args["kind"]) for field in fields},
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_TIMESTAMP(
        cls, scale: DataTypeParam | None = None, timezone: DataTypeParam | None = None
    ) -> dt.Timestamp:
        return dt.Timestamp(
            timezone=timezone if timezone is None else timezone.this.this,
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_TIMESTAMPTZ(cls, scale: DataTypeParam | None = None) -> dt.Timestamp:
        return dt.Timestamp(
            timezone="UTC",
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_DATETIME(cls, scale: DataTypeParam | None = None) -> dt.Timestamp:
        return dt.Timestamp(
            timezone="UTC",
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_INTERVAL(cls, precision: DataTypeParam | None = None) -> dt.Interval:
        if precision is None:
            precision = cls.default_interval_precision
        return dt.Interval(str(precision), nullable=cls.default_nullable)

    @classmethod
    def _get_DECIMAL(
        cls, precision: DataTypeParam | None = None, scale: DataTypeParam | None = None
    ) -> dt.Decimal:
        if precision is None:
            precision = cls.default_decimal_precision
        else:
            precision = int(precision.this.this)

        if scale is None:
            scale = cls.default_decimal_scale
        else:
            scale = int(scale.this.this)

        return dt.Decimal(precision, scale, nullable=cls.default_nullable)

    @classmethod
    def _get_type(cls, parse_result: DataType) -> dt.DataType:
        typ = parse_result.this

        if (result := SQLGLOT_TYPE_TO_IBIS_TYPE.get(typ)) is not None:
            return result.copy(nullable=cls.default_nullable)
        elif (method := getattr(cls, f"_get_{typ.name}", None)) is not None:
            return method(*parse_result.expressions)
        else:
            raise exc.IbisTypeError(f"Unknown type: {typ}")
