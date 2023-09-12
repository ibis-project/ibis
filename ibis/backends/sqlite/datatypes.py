"""Parse SQLite data types."""

from __future__ import annotations

import datetime

import sqlalchemy.types as sat
from sqlalchemy.dialects import sqlite

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import SQLiteType as SqlglotSQLiteType


class SqliteType(AlchemyType):
    dialect = "sqlite"

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sat.TypeEngine:
        if dtype.is_floating():
            return sat.REAL
        else:
            return super().from_ibis(dtype)

    @classmethod
    def to_ibis(cls, typ: sat.TypeEngine, nullable: bool = True) -> dt.DataType:
        if isinstance(typ, sat.REAL):
            return dt.Float64(nullable=nullable)
        elif isinstance(typ, sqlite.JSON):
            return dt.JSON(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotSQLiteType.from_string(type_string, nullable=nullable)


class ISODATETIME(sqlite.DATETIME):
    """A thin `datetime` type to override sqlalchemy's datetime parsing.

    This is to support a wider range of timestamp formats accepted by SQLite.

    See https://sqlite.org/lang_datefunc.html#time_values for the full
    list of datetime formats SQLite accepts.
    """

    def result_processor(self, *_):
        def process(value: str | None) -> datetime.datetime | None:
            """Convert a `str` to a `datetime` according to SQLite's rules.

            This function ignores `None` values.
            """
            if value is None:
                return None
            if value.endswith("Z"):
                # Parse and set the timezone as UTC
                o = datetime.datetime.fromisoformat(value[:-1]).replace(
                    tzinfo=datetime.timezone.utc
                )
            else:
                o = datetime.datetime.fromisoformat(value)
                if o.tzinfo:
                    # Convert any aware datetime to UTC
                    return o.astimezone(datetime.timezone.utc)
            return o

        return process
