from __future__ import annotations

from functools import partial

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.common.collections import FrozenDict
from ibis.formats import TypeMapper

typecode = sge.DataType.Type

_from_sqlglot_types = {
    typecode.BIGDECIMAL: partial(dt.Decimal, 76, 38),
    typecode.BIGINT: dt.Int64,
    typecode.BINARY: dt.Binary,
    typecode.BOOLEAN: dt.Boolean,
    typecode.CHAR: dt.String,
    typecode.DATE: dt.Date,
    typecode.DOUBLE: dt.Float64,
    typecode.ENUM: dt.String,
    typecode.ENUM8: dt.String,
    typecode.ENUM16: dt.String,
    typecode.FLOAT: dt.Float32,
    typecode.FIXEDSTRING: dt.String,
    typecode.GEOMETRY: partial(dt.GeoSpatial, geotype="geometry"),
    typecode.GEOGRAPHY: partial(dt.GeoSpatial, geotype="geography"),
    typecode.HSTORE: partial(dt.Map, dt.string, dt.string),
    typecode.INET: dt.INET,
    typecode.INT128: partial(dt.Decimal, 38, 0),
    typecode.INT256: partial(dt.Decimal, 76, 0),
    typecode.INT: dt.Int32,
    typecode.IPADDRESS: dt.INET,
    typecode.JSON: dt.JSON,
    typecode.JSONB: dt.JSON,
    typecode.LONGBLOB: dt.Binary,
    typecode.LONGTEXT: dt.String,
    typecode.MEDIUMBLOB: dt.Binary,
    typecode.MEDIUMTEXT: dt.String,
    typecode.MONEY: dt.Int64,
    typecode.NCHAR: dt.String,
    typecode.UUID: dt.UUID,
    typecode.NULL: dt.Null,
    typecode.NVARCHAR: dt.String,
    typecode.OBJECT: partial(dt.Map, dt.string, dt.json),
    typecode.SMALLINT: dt.Int16,
    typecode.SMALLMONEY: dt.Int32,
    typecode.TEXT: dt.String,
    typecode.TIME: dt.Time,
    typecode.TIMETZ: dt.Time,
    typecode.TINYBLOB: dt.Binary,
    typecode.TINYINT: dt.Int8,
    typecode.TINYTEXT: dt.String,
    typecode.UBIGINT: dt.UInt64,
    typecode.UINT: dt.UInt32,
    typecode.USMALLINT: dt.UInt16,
    typecode.UTINYINT: dt.UInt8,
    typecode.UUID: dt.UUID,
    typecode.VARBINARY: dt.Binary,
    typecode.VARCHAR: dt.String,
    typecode.VARIANT: dt.JSON,
    typecode.UNIQUEIDENTIFIER: dt.UUID,
    typecode.SET: partial(dt.Array, dt.string),
    #############################
    # Unsupported sqlglot types #
    #############################
    # BIT = auto() # mysql
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

_to_sqlglot_types = {
    dt.Null: typecode.NULL,
    dt.Boolean: typecode.BOOLEAN,
    dt.Int8: typecode.TINYINT,
    dt.Int16: typecode.SMALLINT,
    dt.Int32: typecode.INT,
    dt.Int64: typecode.BIGINT,
    dt.UInt8: typecode.UTINYINT,
    dt.UInt16: typecode.USMALLINT,
    dt.UInt32: typecode.UINT,
    dt.UInt64: typecode.UBIGINT,
    dt.Float16: typecode.FLOAT,
    dt.Float32: typecode.FLOAT,
    dt.Float64: typecode.DOUBLE,
    dt.String: typecode.VARCHAR,
    dt.Binary: typecode.VARBINARY,
    dt.JSON: typecode.JSON,
    dt.INET: typecode.INET,
    dt.UUID: typecode.UUID,
    dt.MACADDR: typecode.VARCHAR,
    dt.Date: typecode.DATE,
    dt.Time: typecode.TIME,
}


class SqlglotType(TypeMapper):
    dialect: str | None = None
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

    unknown_type_strings: dict[str, dt.DataType] = {}
    """String to ibis datatype mapping to use when converting unknown types."""

    @classmethod
    def to_ibis(cls, typ: sge.DataType, nullable: bool | None = None) -> dt.DataType:
        """Convert a sqlglot type to an ibis type."""
        typecode = typ.this

        # broken sqlglot thing
        if isinstance(typecode, sge.Interval):
            typ = sge.DataType(
                this=sge.DataType.Type.INTERVAL,
                expressions=[sge.IntervalSpan(this=typecode.unit)],
            )
            typecode = typ.this

        if method := getattr(cls, f"_from_sqlglot_{typecode.name}", None):
            dtype = method(*typ.expressions)
        else:
            dtype = _from_sqlglot_types[typecode](nullable=cls.default_nullable)

        if nullable is not None:
            return dtype.copy(nullable=nullable)
        else:
            return dtype

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sge.DataType:
        """Convert a sqlglot type to an ibis type."""

        if method := getattr(cls, f"_from_ibis_{dtype.name}", None):
            return method(dtype)
        else:
            return sge.DataType(this=_to_sqlglot_types[type(dtype)])

    @classmethod
    def from_string(cls, text: str, nullable: bool | None = None) -> dt.DataType:
        if dtype := cls.unknown_type_strings.get(text.lower()):
            return dtype

        sgtype = sg.parse_one(text, into=sge.DataType, read=cls.dialect)
        return cls.to_ibis(sgtype, nullable=nullable)

    @classmethod
    def to_string(cls, dtype: dt.DataType) -> str:
        return cls.from_ibis(dtype).sql(dialect=cls.dialect)

    @classmethod
    def _from_sqlglot_ARRAY(cls, value_type: sge.DataType) -> dt.Array:
        return dt.Array(cls.to_ibis(value_type), nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_MAP(
        cls, key_type: sge.DataType, value_type: sge.DataType
    ) -> dt.Map:
        return dt.Map(
            cls.to_ibis(key_type),
            cls.to_ibis(value_type),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_STRUCT(cls, *fields: sge.ColumnDef) -> dt.Struct:
        types = {}
        for i, field in enumerate(fields):
            if isinstance(field, sge.ColumnDef):
                types[field.name] = cls.to_ibis(field.args["kind"])
            else:
                types[f"f{i:d}"] = cls.from_string(str(field))
        return dt.Struct(types, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP(cls, scale=None) -> dt.Timestamp:
        return dt.Timestamp(
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_TIMESTAMPTZ(cls, scale=None) -> dt.Timestamp:
        return dt.Timestamp(
            timezone="UTC",
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_TIMESTAMPLTZ(cls, scale=None) -> dt.Timestamp:
        return dt.Timestamp(
            timezone="UTC",
            scale=cls.default_temporal_scale if scale is None else int(scale.this.this),
            nullable=cls.default_nullable,
        )

    @classmethod
    def _from_sqlglot_INTERVAL(
        cls, precision_or_span: sge.IntervalSpan | None = None
    ) -> dt.Interval:
        nullable = cls.default_nullable
        if precision_or_span is None:
            precision_or_span = cls.default_interval_precision

        if isinstance(precision_or_span, str):
            return dt.Interval(precision_or_span, nullable=nullable)
        elif isinstance(precision_or_span, sge.IntervalSpan):
            if (expression := precision_or_span.expression) is not None:
                unit = expression.this
            else:
                unit = precision_or_span.this.this
            return dt.Interval(unit=unit, nullable=nullable)
        elif precision_or_span is None:
            raise com.IbisTypeError("Interval precision is None")
        else:
            raise com.IbisTypeError(precision_or_span)

    @classmethod
    def _from_sqlglot_DECIMAL(
        cls,
        precision: sge.DataTypeParam | None = None,
        scale: sge.DataTypeParam | None = None,
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
    def _from_sqlglot_GEOMETRY(
        cls, arg: sge.DataTypeParam | None = None
    ) -> sge.DataType:
        if arg is not None:
            return getattr(dt, str(arg))(nullable=cls.default_nullable)
        return dt.GeoSpatial(geotype="geometry", nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_GEOGRAPHY(cls) -> sge.DataType:
        return dt.GeoSpatial(geotype="geography", nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_Interval(cls, dtype: dt.Interval) -> sge.DataType:
        assert dtype.unit is not None, "interval unit cannot be None"
        return sge.DataType(
            this=typecode.INTERVAL,
            expressions=[sge.Var(this=dtype.unit.name)],
        )

    @classmethod
    def _from_ibis_Array(cls, dtype: dt.Array) -> sge.DataType:
        value_type = cls.from_ibis(dtype.value_type)
        return sge.DataType(this=typecode.ARRAY, expressions=[value_type], nested=True)

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        key_type = cls.from_ibis(dtype.key_type)
        value_type = cls.from_ibis(dtype.value_type)
        return sge.DataType(
            this=typecode.MAP, expressions=[key_type, value_type], nested=True
        )

    @classmethod
    def _from_ibis_Struct(cls, dtype: dt.Struct) -> sge.DataType:
        fields = [
            sge.ColumnDef(this=str(name), kind=cls.from_ibis(field))
            for name, field in dtype.items()
        ]
        return sge.DataType(this=typecode.STRUCT, expressions=fields, nested=True)

    @classmethod
    def _from_ibis_Decimal(cls, dtype: dt.Decimal) -> sge.DataType:
        if (precision := dtype.precision) is None:
            precision = cls.default_decimal_precision

        if (scale := dtype.scale) is None:
            scale = cls.default_decimal_scale

        expressions = []

        if precision is not None:
            expressions.append(sge.DataTypeParam(this=sge.Literal.number(precision)))

        if scale is not None:
            if precision is None:
                raise com.IbisTypeError(
                    "Decimal scale cannot be specified without precision"
                )
            expressions.append(sge.DataTypeParam(this=sge.Literal.number(scale)))

        return sge.DataType(this=typecode.DECIMAL, expressions=expressions or None)

    @classmethod
    def _from_ibis_Timestamp(cls, dtype: dt.Timestamp) -> sge.DataType:
        code = typecode.TIMESTAMP if dtype.timezone is None else typecode.TIMESTAMPTZ
        if dtype.scale is not None:
            scale = sge.DataTypeParam(this=sge.Literal.number(dtype.scale))
            return sge.DataType(this=code, expressions=[scale])
        else:
            return sge.DataType(this=code)

    @classmethod
    def _from_ibis_GeoSpatial(cls, dtype: dt.GeoSpatial):
        if (geotype := dtype.geotype) is not None:
            return sge.DataType(this=getattr(typecode, geotype.upper()))
        return sge.DataType(this=typecode.GEOMETRY)

    _from_ibis_Point = (
        _from_ibis_LineString
    ) = (
        _from_ibis_Polygon
    ) = (
        _from_ibis_MultiLineString
    ) = _from_ibis_MultiPoint = _from_ibis_MultiPolygon = _from_ibis_GeoSpatial


class PostgresType(SqlglotType):
    dialect = "postgres"
    default_interval_precision = "s"
    default_temporal_scale = 6

    unknown_type_strings = FrozenDict(
        {
            "vector": dt.unknown,
            "tsvector": dt.unknown,
            "line": dt.linestring,
            "line[]": dt.Array(dt.linestring),
            "polygon": dt.polygon,
            "polygon[]": dt.Array(dt.polygon),
            "point": dt.point,
            "point[]": dt.Array(dt.point),
            "macaddr": dt.macaddr,
            "macaddr[]": dt.Array(dt.macaddr),
            "macaddr8": dt.macaddr,
            "macaddr8[]": dt.Array(dt.macaddr),
        }
    )

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        if not dtype.key_type.is_string():
            raise com.IbisTypeError("Postgres only supports string keys in maps")
        if not dtype.value_type.is_string():
            raise com.IbisTypeError("Postgres only supports string values in maps")
        return sge.DataType(this=typecode.HSTORE)


class DataFusionType(PostgresType):
    unknown_type_strings = {
        "utf8": dt.string,
        "float64": dt.float64,
    }


class MySQLType(SqlglotType):
    dialect = "mysql"
    # these are mysql's defaults, see
    # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
    default_decimal_precision = 10
    default_decimal_scale = 0

    unknown_type_strings = FrozenDict(
        {
            "year(4)": dt.int8,
            "inet6": dt.inet,
        }
    )

    @classmethod
    def _from_sqlglot_BIT(cls, nbits: sge.DataTypeParam) -> dt.Integer:
        nbits = int(nbits.this.this)
        if nbits > 32:
            return dt.Int64(nullable=cls.default_nullable)
        elif nbits > 16:
            return dt.Int32(nullable=cls.default_nullable)
        elif nbits > 8:
            return dt.Int16(nullable=cls.default_nullable)
        else:
            return dt.Int8(nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_DATETIME(cls) -> dt.Timestamp:
        return dt.Timestamp(nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP(cls) -> dt.Timestamp:
        return dt.Timestamp(timezone="UTC", nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_String(cls, dtype: dt.String) -> sge.DataType:
        return sge.DataType(this=typecode.TEXT)


class DuckDBType(SqlglotType):
    dialect = "duckdb"
    default_decimal_precision = 18
    default_decimal_scale = 3
    default_interval_precision = "us"

    unknown_type_strings = FrozenDict({"wkb_blob": dt.binary})

    @classmethod
    def _from_sqlglot_TIMESTAMP(cls) -> dt.Timestamp:
        return dt.Timestamp(scale=6, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMPTZ(cls) -> dt.Timestamp:
        return dt.Timestamp(scale=6, timezone="UTC", nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP_S(cls) -> dt.Timestamp:
        return dt.Timestamp(scale=0, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP_MS(cls) -> dt.Timestamp:
        return dt.Timestamp(scale=3, nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_TIMESTAMP_NS(cls) -> dt.Timestamp:
        return dt.Timestamp(scale=9, nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_GeoSpatial(cls, dtype: dt.GeoSpatial):
        assert (
            dtype.geotype == "geometry"
        ), "DuckDB only supports geometry types; geography types are not supported"
        return sge.DataType(this=typecode.GEOMETRY)


class TrinoType(SqlglotType):
    dialect = "trino"
    default_decimal_precision = 18
    default_decimal_scale = 3
    default_temporal_scale = 3

    unknown_type_strings = FrozenDict(
        {
            "interval year to month": dt.Interval("M"),
            "interval day to second": dt.Interval("ms"),
        }
    )

    @classmethod
    def _from_ibis_Interval(cls, dtype: dt.Interval) -> sge.DataType:
        assert dtype.unit is not None, "interval unit cannot be None"
        if (short := dtype.unit.short) in ("Y", "Q", "M"):
            return sge.DataType(
                this=typecode.INTERVAL,
                expressions=[
                    sge.IntervalSpan(
                        this=sge.Var(this="YEAR"), expression=sge.Var(this="MONTH")
                    )
                ],
            )
        elif short in ("D", "h", "m", "s", "ms", "us", "ns"):
            return sge.DataType(
                this=typecode.INTERVAL,
                expressions=[
                    sge.IntervalSpan(
                        this=sge.Var(this="DAY"), expression=sge.Var(this="SECOND")
                    )
                ],
            )
        else:
            raise NotImplementedError(
                f"Trino does not support {dtype.unit.name} intervals"
            )

    @classmethod
    def _from_sqlglot_UBIGINT(cls):
        return dt.Decimal(precision=19, scale=0, nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_UInt64(cls, dtype):
        return sge.DataType(
            this=typecode.DECIMAL,
            expressions=[
                sge.DataTypeParam(this=sge.convert(19)),
                sge.DataTypeParam(this=sge.convert(0)),
            ],
        )

    @classmethod
    def _from_sqlglot_UINT(cls):
        return dt.Int64(nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_UInt32(cls, dtype):
        return sge.DataType(this=typecode.BIGINT)

    @classmethod
    def _from_sqlglot_USMALLINT(cls):
        return dt.Int32(nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_UInt16(cls, dtype):
        return sge.DataType(this=typecode.INT)

    @classmethod
    def _from_sqlglot_UTINYINT(cls):
        return dt.Int16(nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_UInt8(cls, dtype):
        return sge.DataType(this=typecode.SMALLINT)


class DruidType(SqlglotType):
    # druid doesn't have a sophisticated type system and hive is close enough
    dialect = "hive"
    unknown_type_strings = FrozenDict({"complex<json>": dt.json})


class OracleType(SqlglotType):
    dialect = "oracle"


class SnowflakeType(SqlglotType):
    dialect = "snowflake"

    default_decimal_precision = 38
    default_decimal_scale = 9

    default_temporal_scale = 9

    @classmethod
    def _from_sqlglot_FLOAT(cls) -> dt.Float64:
        return dt.Float64(nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_DECIMAL(cls, precision=None, scale=None) -> dt.Decimal:
        if scale is None or int(scale.this.this) == 0:
            return dt.Int64(nullable=cls.default_nullable)
        else:
            return super()._from_sqlglot_DECIMAL(precision, scale)

    @classmethod
    def _from_sqlglot_ARRAY(cls, value_type=None) -> dt.Array:
        assert value_type is None
        return dt.Array(dt.json, nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_JSON(cls, dtype: dt.JSON) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.VARIANT)

    @classmethod
    def _from_ibis_Array(cls, dtype: dt.Array) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.ARRAY, nested=True)

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.OBJECT, nested=True)

    @classmethod
    def _from_ibis_Struct(cls, dtype: dt.Struct) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.OBJECT, nested=True)


class SQLiteType(SqlglotType):
    dialect = "sqlite"


class PySparkType(SqlglotType):
    dialect = "spark"
    default_decimal_precision = 38
    default_decimal_scale = 18
