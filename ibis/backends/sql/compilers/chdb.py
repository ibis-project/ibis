from __future__ import annotations

import sqlglot.expressions as sge

from ibis.backends.sql.compilers.clickhouse import ClickHouseCompiler


class ChdbCompiler(ClickHouseCompiler):
    """ClickHouse compiler that renders in-memory tables as ``Python(name)``."""

    __slots__ = ()

    def visit_InMemoryTable(self, op, *, name, schema, data):
        return sge.Table(this=self.f.Python(sge.convert(name)))


compiler = ChdbCompiler()
