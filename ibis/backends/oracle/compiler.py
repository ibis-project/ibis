from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import OracleType
from ibis.expr.rewrites import rewrite_sample


@public
class OracleCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "oracle"
    type_mapper = OracleType
    rewrites = (rewrite_sample, *SQLGlotCompiler.rewrites)

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)
