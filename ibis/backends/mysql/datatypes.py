from __future__ import annotations

import sqlalchemy.types as sat
from sqlalchemy.dialects import mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import UUID, AlchemyType
from ibis.backends.base.sqlglot.datatypes import MySQLType as SqlglotMySQLType


class MySQLDateTime(mysql.DATETIME):
    """Custom DATETIME type for MySQL that handles zero values."""

    def result_processor(self, *_):
        return lambda v: None if v == "0000-00-00 00:00:00" else v


_to_mysql_types = {
    dt.Boolean: mysql.BOOLEAN,
    dt.Int8: mysql.TINYINT,
    dt.Int16: mysql.SMALLINT,
    dt.Int32: mysql.INTEGER,
    dt.Int64: mysql.BIGINT,
    dt.Float16: mysql.FLOAT,
    dt.Float32: mysql.FLOAT,
    dt.Float64: mysql.DOUBLE,
    dt.String: mysql.TEXT,
    dt.JSON: mysql.JSON,
    dt.Timestamp: MySQLDateTime,
}

_from_mysql_types = {
    mysql.BIGINT: dt.Int64,
    mysql.BINARY: dt.Binary,
    mysql.BLOB: dt.Binary,
    mysql.BOOLEAN: dt.Boolean,
    mysql.DATETIME: dt.Timestamp,
    mysql.DOUBLE: dt.Float64,
    mysql.FLOAT: dt.Float32,
    mysql.INTEGER: dt.Int32,
    mysql.JSON: dt.JSON,
    mysql.LONGBLOB: dt.Binary,
    mysql.LONGTEXT: dt.String,
    mysql.MEDIUMBLOB: dt.Binary,
    mysql.MEDIUMINT: dt.Int32,
    mysql.MEDIUMTEXT: dt.String,
    mysql.REAL: dt.Float64,
    mysql.SMALLINT: dt.Int16,
    mysql.TEXT: dt.String,
    mysql.DATE: dt.Date,
    mysql.TINYBLOB: dt.Binary,
    mysql.TINYINT: dt.Int8,
    mysql.VARBINARY: dt.Binary,
    mysql.VARCHAR: dt.String,
    mysql.ENUM: dt.String,
    mysql.CHAR: dt.String,
    mysql.TIME: dt.Time,
    mysql.YEAR: dt.Int8,
    MySQLDateTime: dt.Timestamp,
    UUID: dt.UUID,
}


class MySQLType(AlchemyType):
    dialect = "mysql"

    @classmethod
    def from_ibis(cls, dtype):
        try:
            return _to_mysql_types[type(dtype)]
        except KeyError:
            return super().from_ibis(dtype)

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if isinstance(typ, (sat.NUMERIC, mysql.NUMERIC, mysql.DECIMAL)):
            # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
            return dt.Decimal(typ.precision or 10, typ.scale or 0, nullable=nullable)
        elif isinstance(typ, mysql.BIT):
            if 1 <= (length := typ.length) <= 8:
                return dt.Int8(nullable=nullable)
            elif 9 <= length <= 16:
                return dt.Int16(nullable=nullable)
            elif 17 <= length <= 32:
                return dt.Int32(nullable=nullable)
            elif 33 <= length <= 64:
                return dt.Int64(nullable=nullable)
            else:
                raise ValueError(f"Invalid MySQL BIT length: {length:d}")
        elif isinstance(typ, mysql.TIMESTAMP):
            return dt.Timestamp(timezone="UTC", nullable=nullable)
        elif isinstance(typ, mysql.SET):
            return dt.Array(dt.string, nullable=nullable)
        elif dtype := _from_mysql_types.get(type(typ)):
            return dtype(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotMySQLType.from_string(type_string, nullable=nullable)
