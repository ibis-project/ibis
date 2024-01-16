from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from sqlglot import exp
from sqlglot.dialects import Postgres
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import DruidType
from ibis.expr.rewrites import rewrite_sample


# Is postgres the best dialect to inherit from?
class Druid(Postgres):
    """The druid dialect."""

    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            exp.ApproxDistinct: rename_func("approx_count_distinct"),
            exp.Pow: rename_func("power"),
        }


class DruidCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "druid"
    type_mapper = DruidType
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

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.concat(*toolz.interpose(sep, arg))

    @visit_node.register(ops.Pi)
    def visit_Pi(self, op):
        return self.f.acos(-1)

    @visit_node.register(ops.Sign)
    def visit_Sign(self, op, *, arg):
        return self.if_(arg.eq(0), 0, self.if_(arg > 0, 1, -1))

    @visit_node.register(ops.GroupConcat)
    def visit_GroupConcat(self, op, *, arg, sep, where):
        return self.agg.string_agg(arg, sep, 1 << 20, where=where)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return self.f.left(arg, self.f.length(start)).eq(start)

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    @visit_node.register(ops.Capitalize)
    def visit_Capitalize(self, op, *, arg):
        return self.if_(
            self.f.length(arg) < 2,
            self.f.upper(arg),
            self.f.concat(
                self.f.upper(self.f.substr(arg, 1, 1)),
                self.f.lower(self.f.substr(arg, 2)),
            ),
        )

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern):
        return self.f.anon.regexp_like(arg, pattern)

    @visit_node.register(ops.StringSQLILike)
    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        if escape is not None:
            raise NotImplementedError("non-None escape not supported")
        return self.f.upper(arg).like(self.f.upper(pattern))

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        if value is None:
            return NULL
        return super().visit_Literal(op, value=value, dtype=dtype)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_uuid():
            return sge.convert(str(value))

        return None

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_integer() and to.is_timestamp():
            # seconds since UNIX epoch
            return self.f.millis_to_timestamp(arg * 1_000)
        elif from_.is_string() and to.is_timestamp():
            return self.f.time_parse(arg)
        return super().visit_Cast(op, arg=arg, to=to)

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.f.time_parse(
            self.f.concat(
                self.f.lpad(self.cast(year, dt.string), 4, "0"),
                "-",
                self.f.lpad(self.cast(month, dt.string), 2, "0"),
                "-",
                self.f.lpad(self.cast(day, dt.string), 2, "0"),
                "T",
                self.f.lpad(self.cast(hours, dt.string), 2, "0"),
                ":",
                self.f.lpad(self.cast(minutes, dt.string), 2, "0"),
                ":",
                self.f.lpad(self.cast(seconds, dt.string), 2, "0"),
                "Z",
            )
        )

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
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.BitwiseAnd: "bitwise_and",
    ops.BitwiseNot: "bitwise_complement",
    ops.BitwiseOr: "bitwise_or",
    ops.BitwiseXor: "bitwise_xor",
    ops.BitwiseLeftShift: "bitwise_shift_left",
    ops.BitwiseRightShift: "bitwise_shift_right",
    ops.Modulus: "mod",
    ops.Power: "power",
    ops.Log10: "log10",
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.StringContains: "contains_string",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @DruidCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @DruidCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(DruidCompiler, f"visit_{_op.__name__}", _fmt)
