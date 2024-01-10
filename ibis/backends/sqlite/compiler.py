from __future__ import annotations

from functools import singledispatchmethod

import sqlglot.expressions as sge
from public import public

from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import SQLiteType


@public
class SQLiteCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "sqlite"
    type_mapper = SQLiteType

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)
