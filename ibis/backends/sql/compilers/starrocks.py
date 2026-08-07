from __future__ import annotations

from sqlglot.dialects import StarRocks

from ibis.backends.sql.compilers.mysql import MySQLCompiler
from ibis.backends.sql.datatypes import StarRocksType


class StarRocksCompiler(MySQLCompiler):
    __slots__ = ()

    dialect = StarRocks
    type_mapper = StarRocksType


compiler = StarRocksCompiler()
