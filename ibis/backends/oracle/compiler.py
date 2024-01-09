from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import OracleType
from ibis.backends.base.sqlglot.rewrites import replace_log2, replace_log10
from ibis.expr.rewrites import rewrite_sample


@public
class OracleCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "oracle"
    quoted = True
    type_mapper = OracleType
    rewrites = (
        rewrite_sample,
        replace_log2,
        replace_log10,
        *SQLGlotCompiler.rewrites,
    )

    NAN = sge.Literal.number("binary_double_nan")
    """Backend's NaN literal."""

    POS_INF = sge.Literal.number("binary_double_infinity")
    """Backend's positive infinity literal."""

    NEG_INF = sge.Literal.number("-binary_double_infinity")
    """Backend's negative infinity literal."""

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.NAN)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        # Not using FuncGen here because of dotted function call
        return sg.func("dbms_random.value")

    @visit_node.register(ops.Pi)
    def visit_Pi(self, op):
        return self.f.acos(-1)

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1 / self.f.tan(arg)

    @visit_node.register(ops.Degrees)
    def visit_Degrees(self, op, *, arg):
        return 180 * arg / self.visit_node(ops.Pi())

    @visit_node.register(ops.Radians)
    def visit_Radians(self, op, *, arg):
        return self.visit_node(ops.Pi()) * arg / 180
