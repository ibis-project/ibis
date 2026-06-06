from __future__ import annotations

import sqlglot as sg
from sqlglot.dialects import StarRocks

from ibis.backends.sql.compilers.starrocks import StarRocksCompiler
from ibis.backends.sql.datatypes import StarRocksType


def test_compiler_uses_sqlglot_starrocks_dialect():
    compiler = StarRocksCompiler()

    assert compiler.dialect is StarRocks
    assert compiler.type_mapper is StarRocksType


def test_sqlglot_starrocks_dialect_is_used_for_generation():
    query = sg.select("a").from_("t").limit(1)

    assert query.sql(StarRocks) == "SELECT a FROM t LIMIT 1"
