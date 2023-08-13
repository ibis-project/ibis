from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
import sqlalchemy.types as sat
from dateutil.parser import parse as timestamp_parse
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.common.collections import FrozenDict
from ibis.formats.parser import TypeParser

if TYPE_CHECKING:
    from collections.abc import Mapping


class DruidDateTime(sat.TypeDecorator):
    impl = sa.TIMESTAMP

    cache_ok = True

    def process_result_value(self, value, dialect):
        return None if value is None else timestamp_parse(value)


class DruidBinary(sa.LargeBinary):
    def result_processor(self, dialect, coltype):
        def process(value):
            return None if value is None else value.encode("utf-8")

        return process


class DruidString(sat.TypeDecorator):
    impl = sa.String

    cache_ok = True

    def process_result_value(self, value, dialect):
        return value


@compiles(sa.BIGINT, "druid")
@compiles(sa.BigInteger, "druid")
def _bigint(element, compiler, **kw):
    return "BIGINT"


@compiles(sa.INTEGER, "druid")
@compiles(sa.Integer, "druid")
def _integer(element, compiler, **kw):
    return "INTEGER"


@compiles(sa.SMALLINT, "druid")
@compiles(sa.SmallInteger, "druid")
def _smallint(element, compiler, **kw):
    return "SMALLINT"


class DruidTypeParser(TypeParser):
    __slots__ = ()

    # druid doesn't have a sophisticated type system and hive is close enough
    dialect = "hive"
    short_circuit: Mapping[str, dt.DataType] = FrozenDict({"complex<json>": dt.json})


parse = DruidTypeParser.parse


class DruidType(AlchemyType):
    dialect = "hive"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if isinstance(typ, DruidDateTime):
            return dt.Timestamp(nullable=nullable)
        elif isinstance(typ, DruidBinary):
            return dt.Binary(nullable=nullable)
        elif isinstance(typ, DruidString):
            return dt.String(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if dtype.is_timestamp():
            return DruidDateTime()
        elif dtype.is_binary():
            return DruidBinary()
        elif dtype.is_string():
            return DruidString()
        else:
            return super().from_ibis(dtype)
