from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import Postgres

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import ExasolType
from ibis.expr.rewrites import rewrite_sample


# Is postgres the best dialect to inherit from?
class Exasol(Postgres):
    """The exasol dialect."""

    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {}


class ExasolCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "exasol"
    type_mapper = ExasolType
    quoted = True
    rewrites = (rewrite_sample, *SQLGlotCompiler.rewrites)

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.InMemoryTable)
    def visit_InMemoryTable(self, op, *, name, schema, data):
        # the performance of this is rather terrible
        tuples = data.to_frame().itertuples(index=False)
        quoted = self.quoted
        columns = [sg.column(col, quoted=quoted) for col in schema.names]
        expr = sge.Values(
            expressions=[
                sge.Tuple(expressions=tuple(map(sge.convert, row))) for row in tuples
            ],
            alias=sge.TableAlias(
                this=sg.to_identifier(name, quoted=quoted),
                columns=columns,
            ),
        )
        return sg.select(*columns).from_(expr)

    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayDistinct)
    @visit_node.register(ops.ArrayFilter)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayIntersect)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayUnion)
    @visit_node.register(ops.ArrayZip)
    @visit_node.register(ops.CountDistinctStar)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.DayOfWeekIndex)
    @visit_node.register(ops.DayOfWeekName)
    @visit_node.register(ops.First)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.IsNan)
    @visit_node.register(ops.IsInf)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.RegexReplace)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.StandardDev)
    @visit_node.register(ops.Strftime)
    @visit_node.register(ops.StringAscii)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.TimestampNow)
    @visit_node.register(ops.Translate)
    @visit_node.register(ops.TypeOf)
    @visit_node.register(ops.Unnest)
    @visit_node.register(ops.Variance)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.Modulus: "mod",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @ExasolCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @ExasolCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(ExasolCompiler, f"visit_{_op.__name__}", _fmt)
