from __future__ import annotations

from functools import partial

import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects.mysql.base import MySQLDialect

import ibis.expr.datatypes as dt

# binary character set
# used to distinguish blob binary vs blob text
MY_CHARSET_BIN = 63


def _type_from_cursor_info(descr, field) -> dt.DataType:
    """Construct an ibis type from MySQL field descr and field result metadata.

    This method is complex because the MySQL protocol is complex.

    Types are not encoded in a self contained way, meaning you need
    multiple pieces of information coming from the result set metadata to
    determine the most precise type for a field. Even then, the decoding is
    not high fidelity in some cases: UUIDs for example are decoded as
    strings, because the protocol does not appear to preserve the logical
    type, only the physical type.
    """
    from pymysql.connections import TEXT_TYPES

    _, type_code, _, _, field_length, scale, _ = descr
    flags = _FieldFlags(field.flags)
    typename = _type_codes.get(type_code)
    if typename is None:
        raise NotImplementedError(f"MySQL type code {type_code:d} is not supported")

    if typename in ("DECIMAL", "NEWDECIMAL"):
        precision = _decimal_length_to_precision(
            length=field_length,
            scale=scale,
            is_unsigned=flags.is_unsigned,
        )
        typ = partial(_type_mapping[typename], precision=precision, scale=scale)
    elif typename == "BIT":
        if field_length <= 8:
            typ = dt.int8
        elif field_length <= 16:
            typ = dt.int16
        elif field_length <= 32:
            typ = dt.int32
        elif field_length <= 64:
            typ = dt.int64
        else:
            raise AssertionError('invalid field length for BIT type')
    elif flags.is_set:
        # sets are limited to strings
        typ = dt.Set(dt.string)
    elif flags.is_unsigned and flags.is_num:
        typ = getattr(dt, f"U{typ.__name__}")
    elif type_code in TEXT_TYPES:
        # binary text
        if field.charsetnr == MY_CHARSET_BIN:
            typ = dt.Binary
        else:
            typ = dt.String
    else:
        typ = _type_mapping[typename]

    # projection columns are always nullable
    return typ(nullable=True)


# ported from my_decimal.h:my_decimal_length_to_precision in mariadb
def _decimal_length_to_precision(*, length: int, scale: int, is_unsigned: bool) -> int:
    return length - (scale > 0) - (not (is_unsigned or not length))


_type_codes = {
    0: "DECIMAL",
    1: "TINY",
    2: "SHORT",
    3: "LONG",
    4: "FLOAT",
    5: "DOUBLE",
    6: "NULL",
    7: "TIMESTAMP",
    8: "LONGLONG",
    9: "INT24",
    10: "DATE",
    11: "TIME",
    12: "DATETIME",
    13: "YEAR",
    15: "VARCHAR",
    16: "BIT",
    245: "JSON",
    246: "NEWDECIMAL",
    247: "ENUM",
    248: "SET",
    249: "TINY_BLOB",
    250: "MEDIUM_BLOB",
    251: "LONG_BLOB",
    252: "BLOB",
    253: "VAR_STRING",
    254: "STRING",
    255: "GEOMETRY",
}


_type_mapping = {
    "DECIMAL": dt.Decimal,
    "TINY": dt.Int8,
    "SHORT": dt.Int16,
    "LONG": dt.Int32,
    "FLOAT": dt.Float32,
    "DOUBLE": dt.Float64,
    "NULL": dt.Null,
    "TIMESTAMP": lambda nullable: dt.Timestamp(timezone="UTC", nullable=nullable),
    "LONGLONG": dt.Int64,
    "INT24": dt.Int32,
    "DATE": dt.Date,
    "TIME": dt.Time,
    "DATETIME": dt.Timestamp,
    "YEAR": dt.Int8,
    "VARCHAR": dt.String,
    "JSON": dt.JSON,
    "NEWDECIMAL": dt.Decimal,
    "ENUM": dt.String,
    "SET": lambda nullable: dt.Set(dt.string, nullable=nullable),
    "TINY_BLOB": dt.Binary,
    "MEDIUM_BLOB": dt.Binary,
    "LONG_BLOB": dt.Binary,
    "BLOB": dt.Binary,
    "VAR_STRING": dt.String,
    "STRING": dt.String,
    "GEOMETRY": dt.Geometry,
}


class _FieldFlags:
    """Flags used to disambiguate field types.

    Gaps in the flag numbers are because we do not map in flags that are
    of no use in determining the field's type, such as whether the field
    is a primary key or not.
    """

    UNSIGNED = 1 << 5
    SET = 1 << 11
    NUM = 1 << 15

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def is_unsigned(self) -> bool:
        return (self.UNSIGNED & self.value) != 0

    @property
    def is_set(self) -> bool:
        return (self.SET & self.value) != 0

    @property
    def is_num(self) -> bool:
        return (self.NUM & self.value) != 0


@dt.dtype.register(MySQLDialect, (sa.NUMERIC, mysql.NUMERIC))
def sa_mysql_numeric(_, satype, nullable=True):
    # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
    return dt.Decimal(satype.precision or 10, satype.scale or 0, nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.YEAR)
@dt.dtype.register(MySQLDialect, mysql.TINYINT)
def sa_mysql_tinyint(_, satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.BIT)
def sa_mysql_bit(_, satype, nullable=True):
    if 1 <= (length := satype.length) <= 8:
        return dt.Int8(nullable=nullable)
    elif 9 <= length <= 16:
        return dt.Int16(nullable=nullable)
    elif 17 <= length <= 32:
        return dt.Int32(nullable=nullable)
    elif 33 <= length <= 64:
        return dt.Int64(nullable=nullable)
    else:
        raise ValueError(f"Invalid MySQL BIT length: {length:d}")


@dt.dtype.register(MySQLDialect, mysql.FLOAT)
def sa_real(_, satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.TIMESTAMP)
def sa_mysql_timestamp(_, satype, nullable=True):
    return dt.Timestamp(timezone="UTC", nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.DATETIME)
def sa_mysql_datetime(_, satype, nullable=True):
    return dt.Timestamp(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.SET)
def sa_mysql_set(_, satype, nullable=True):
    return dt.Set(dt.string, nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.DOUBLE)
def sa_mysql_double(_, satype, nullable=True):
    # TODO: handle asdecimal=True
    return dt.Float64(nullable=nullable)


@dt.dtype.register(
    MySQLDialect,
    (
        mysql.TINYBLOB,
        mysql.MEDIUMBLOB,
        mysql.BLOB,
        mysql.LONGBLOB,
        mysql.BINARY,
        mysql.VARBINARY,
    ),
)
def sa_binary(_, satype, nullable=True):
    return dt.Binary(nullable=nullable)


# TODO(kszucs): unsigned integers


@dt.dtype.register((mysql.DOUBLE, mysql.REAL))
def mysql_double(satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(mysql.FLOAT)
def mysql_float(satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(mysql.TINYINT)
def mysql_tinyint(satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(mysql.BLOB)
def mysql_blob(satype, nullable=True):
    return dt.Binary(nullable=nullable)
