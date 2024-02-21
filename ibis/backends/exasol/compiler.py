from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compiler import NULL, SQLGlotCompiler
from ibis.backends.sql.datatypes import ExasolType
from ibis.backends.sql.dialects import Exasol
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_empty_order_by_window,
    rewrite_sample_as_filter,
)


class ExasolCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Exasol
    type_mapper = ExasolType
    rewrites = (
        rewrite_sample_as_filter,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.AnalyticVectorizedUDF,
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
            ops.ArrayStringJoin,
            ops.ArrayUnion,
            ops.ArrayZip,
            ops.BitwiseNot,
            ops.Covariance,
            ops.CumeDist,
            ops.DateAdd,
            ops.DateDelta,
            ops.DateSub,
            ops.DateFromYMD,
            ops.DayOfWeekIndex,
            ops.DayOfWeekName,
            ops.ElementWiseVectorizedUDF,
            ops.ExtractDayOfYear,
            ops.ExtractEpochSeconds,
            ops.ExtractQuarter,
            ops.ExtractWeekOfYear,
            ops.First,
            ops.IntervalFromInteger,
            ops.IsInf,
            ops.IsNan,
            ops.Last,
            ops.Levenshtein,
            ops.Median,
            ops.MultiQuantile,
            ops.Quantile,
            ops.ReductionVectorizedUDF,
            ops.RegexExtract,
            ops.RegexReplace,
            ops.RegexSearch,
            ops.RegexSplit,
            ops.RowID,
            ops.StandardDev,
            ops.Strftime,
            ops.StringJoin,
            ops.StringSplit,
            ops.StringToTimestamp,
            ops.TimeDelta,
            ops.TimestampAdd,
            ops.TimestampBucket,
            ops.TimestampDelta,
            ops.TimestampDiff,
            ops.TimestampNow,
            ops.TimestampSub,
            ops.TimestampTruncate,
            ops.TypeOf,
            ops.Unnest,
            ops.Variance,
        )
    )

    SIMPLE_OPS = {
        ops.Log10: "log10",
        ops.Modulus: "mod",
        ops.All: "min",
        ops.Any: "max",
    }

    @staticmethod
    def _minimize_spec(start, end, spec):
        if (
            start is None
            and isinstance(getattr(end, "value", None), ops.Literal)
            and end.value.value == 0
            and end.following
        ):
            return None
        return spec

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)
        return func(*args)

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        """Exasol does not allow dots in quoted column names."""
        return name.replace(".", "_")

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_timestamp():
            val = value.replace(tzinfo=None).isoformat(sep=" ", timespec="milliseconds")
            return self.cast(val, dtype)
        elif dtype.is_array() or dtype.is_struct() or dtype.is_map():
            raise com.UnsupportedBackendType(
                f"{type(dtype).__name__}s are not supported in Exasol"
            )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)

    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    def visit_StartsWith(self, op, *, arg, start):
        return self.f.left(arg, self.f.length(start)).eq(start)

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        return self.f.locate(substr, arg, (start if start is not None else 0) + 1)

    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return self.f.upper(arg).like(self.f.upper(pattern))

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.locate(needle, haystack) > 0

    def visit_ExtractSecond(self, op, *, arg):
        return self.f.floor(self.cast(self.f.extract(self.v.second, arg), op.dtype))

    def visit_StringConcat(self, op, *, arg):
        any_args_null = (a.is_(NULL) for a in arg)
        return self.if_(sg.or_(*any_args_null), NULL, self.f.concat(*arg))

    def visit_CountDistinctStar(self, op, *, arg, where):
        raise com.UnsupportedOperationError(
            "COUNT(DISTINCT *) is not supported in Exasol"
        )

    def visit_DateTruncate(self, op, *, arg, unit):
        return super().visit_TimestampTruncate(op, arg=arg, unit=unit)
