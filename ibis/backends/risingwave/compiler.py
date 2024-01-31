from __future__ import annotations

from functools import singledispatchmethod

from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.postgres.compiler import PostgresCompiler


@public
class RisingwaveCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = "postgres"
    name = "risingwave"

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg) for arg in args)
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.name} only implements `pop` correlation coefficient"
            )
        super().visit_Correlation(op, left=left, right=right, how=how, where=where)
