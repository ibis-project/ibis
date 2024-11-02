from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import DruidType
from ibis.backends.sql.dialects import Druid
from ibis.common.temporal import TimestampUnit


class DruidCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Druid
    type_mapper = DruidType

    agg = AggGen(supports_filter=True)

    LOWERED_OPS = {ops.Capitalize: None, ops.Sample: None}

    UNSUPPORTED_OPS = (
        ops.ApproxMedian,
        ops.ArrayDistinct,
        ops.ArrayFilter,
        ops.ArrayFlatten,
        ops.ArrayIntersect,
        ops.ArrayMap,
        ops.ArraySort,
        ops.ArrayUnion,
        ops.ArrayZip,
        ops.ArgMax,
        ops.ArgMin,
        ops.CountDistinctStar,
        ops.Covariance,
        ops.Date,
        ops.DateDelta,
        ops.DateFromYMD,
        ops.DayOfWeekIndex,
        ops.DayOfWeekName,
        ops.IntervalFromInteger,
        ops.IsNan,
        ops.IsInf,
        ops.Levenshtein,
        ops.Median,
        ops.RandomScalar,
        ops.RandomUUID,
        ops.RegexReplace,
        ops.RegexSplit,
        ops.RowID,
        ops.StandardDev,
        ops.Strftime,
        ops.StringAscii,
        ops.StringSplit,
        ops.StringToDate,
        ops.StringToTime,
        ops.StringToTimestamp,
        ops.TimeDelta,
        ops.TimestampBucket,
        ops.TimestampDelta,
        ops.Translate,
        ops.TypeOf,
        ops.Unnest,
        ops.Variance,
        ops.Sample,
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
        ops.Power: "power",
        ops.ApproxCountDistinct: "approx_count_distinct",
        ops.StringContains: "contains_string",
    }

    def visit_Modulus(self, op, *, left, right):
        return self.f.anon.mod(left, right)

    def visit_Log10(self, op, *, arg):
        return self.f.anon.log10(arg)

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

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        return self.agg.string_agg(arg, sep, 1 << 20, where=where, order_by=order_by)

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
            # types that cannot be cast to NULL are null, and temporal types
            # and druid doesn't have a bytes type so don't cast that
            if dtype.is_null() or dtype.is_temporal() or dtype.is_binary():
                return NULL
            else:
                return self.cast(NULL, dtype)
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

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        if unit == TimestampUnit.SECOND:
            return self.f.millis_to_timestamp(arg * 1_000)
        elif unit == TimestampUnit.MILLISECOND:
            return self.f.millis_to_timestamp(arg)
        raise exc.UnsupportedArgumentError(f"Druid doesn't support {unit} units")

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


compiler = DruidCompiler()
