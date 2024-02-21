from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compiler import NULL, SQLGlotCompiler
from ibis.backends.sql.datatypes import DruidType
from ibis.backends.sql.dialects import Druid
from ibis.backends.sql.rewrites import (
    rewrite_capitalize,
    rewrite_sample_as_filter,
)


class DruidCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Druid
    type_mapper = DruidType
    rewrites = (
        rewrite_sample_as_filter,
        *(
            rewrite
            for rewrite in SQLGlotCompiler.rewrites
            if rewrite is not rewrite_capitalize
        ),
    )

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.ApproxMedian,
            ops.Arbitrary,
            ops.ArgMax,
            ops.ArgMin,
            ops.ArrayCollect,
            ops.ArrayDistinct,
            ops.ArrayFilter,
            ops.ArrayFlatten,
            ops.ArrayIntersect,
            ops.ArrayMap,
            ops.ArraySort,
            ops.ArrayUnion,
            ops.ArrayZip,
            ops.CountDistinctStar,
            ops.Covariance,
            ops.DateDelta,
            ops.DayOfWeekIndex,
            ops.DayOfWeekName,
            ops.First,
            ops.IntervalFromInteger,
            ops.IsNan,
            ops.IsInf,
            ops.Last,
            ops.Levenshtein,
            ops.Median,
            ops.MultiQuantile,
            ops.Quantile,
            ops.RegexReplace,
            ops.RegexSplit,
            ops.RowID,
            ops.StandardDev,
            ops.Strftime,
            ops.StringAscii,
            ops.StringSplit,
            ops.StringToTimestamp,
            ops.TimeDelta,
            ops.TimestampBucket,
            ops.TimestampDelta,
            ops.TimestampNow,
            ops.Translate,
            ops.TypeOf,
            ops.Unnest,
            ops.Variance,
        )
    )

    SIMPLE_OPS = {
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

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
        return expr

    def visit_Sum(self, op, *, arg, where):
        arg = self.if_(arg, 1, 0) if op.arg.dtype.is_boolean() else arg
        return self.agg.sum(arg, where=where)

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

    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.concat(*toolz.interpose(sep, arg))

    def visit_Pi(self, op):
        return self.f.acos(-1)

    def visit_Sign(self, op, *, arg):
        return self.if_(arg.eq(0), 0, self.if_(arg > 0, 1, -1))

    def visit_GroupConcat(self, op, *, arg, sep, where):
        return self.agg.string_agg(arg, sep, 1 << 20, where=where)

    def visit_StartsWith(self, op, *, arg, start):
        return self.f.left(arg, self.f.length(start)).eq(start)

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    def visit_Capitalize(self, op, *, arg):
        return self.if_(
            self.f.length(arg) < 2,
            self.f.upper(arg),
            self.f.concat(
                self.f.upper(self.f.substr(arg, 1, 1)),
                self.f.lower(self.f.substr(arg, 2)),
            ),
        )

    def visit_RegexSearch(self, op, *, arg, pattern):
        return self.f.anon.regexp_like(arg, pattern)

    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        if escape is not None:
            raise NotImplementedError("non-None escape not supported")
        return self.f.upper(arg).like(self.f.upper(pattern))

    def visit_Literal(self, op, *, value, dtype):
        if value is None:
            return NULL
        return super().visit_Literal(op, value=value, dtype=dtype)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_uuid():
            return sge.convert(str(value))

        return None

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_integer() and to.is_timestamp():
            # seconds since UNIX epoch
            return self.f.millis_to_timestamp(arg * 1_000)
        elif from_.is_string() and to.is_timestamp():
            return self.f.time_parse(arg)
        return super().visit_Cast(op, arg=arg, to=to)

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
