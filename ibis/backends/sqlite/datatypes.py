"""Parse SQLite data types."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.sqlite.base import SQLiteDialect

import ibis.expr.datatypes as dt


def parse(text: str) -> dt.DataType:
    """Parse `text` into an ibis data type."""
    text = text.strip().upper()

    # SQLite affinity rules
    # (see https://www.sqlite.org/datatype3.html).

    # 1. If the declared type contains the string "INT" then it is
    # assigned INTEGER affinity.
    if "INT" in text:
        return dt.int64

    # 2. If the declared type of the column contains any of the
    # strings "CHAR", "CLOB", or "TEXT" then that column has TEXT
    # affinity. Notice that the type VARCHAR contains the string
    # "CHAR" and is thus assigned TEXT affinity.
    if "CHAR" in text or "CLOB" in text or "TEXT" in text:
        return dt.string

    # 3. If the declared type for a column contains the string "BLOB"
    # or if no type is specified then the column has affinity BLOB.
    if not text or "BLOB" in text:
        return dt.binary

    # 4. If the declared type for a column contains any of the strings
    # "REAL", "FLOA", or "DOUB" then the column has REAL affinity.
    if "REAL" in text or "FLOA" in text or "DOUB" in text:
        return dt.float64

    # 5. Otherwise, the affinity is NUMERIC.
    return dt.decimal


@dt.dtype.register(SQLiteDialect, sqlite.NUMERIC)
def sa_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision, satype.scale, nullable=nullable)


@dt.dtype.register(SQLiteDialect, sa.REAL)
def sa_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)
