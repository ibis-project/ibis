from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR, SQLGlotCompiler
from ibis.backends.sql.datatypes import ExasolType
from ibis.backends.sql.dialects import Exasol
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_empty_order_by_window,
    split_select_distinct_with_order_by,
)


class ExasolCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Exasol
    type_mapper = ExasolType
    rewrites = (
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )

    post_rewrites = (split_select_distinct_with_order_by,)

    UNSUPPORTED_OPS = (
        ops.AnalyticVectorizedUDF,
        ops.ArrayDistinct,
        ops.ArrayFilter,
        ops.ArrayFlatten,
        ops.ArrayIntersect,
        ops.ArrayMap,
        ops.ArraySort,
        ops.ArrayStringJoin,
        ops.ArrayUnion,
        ops.ArrayZip,
        ops.ArgMax,
        ops.ArgMin,
        ops.BitwiseNot,
        ops.CumeDist,
        ops.DateAdd,
        ops.DateSub,
        ops.DateFromYMD,
        ops.DayOfWeekIndex,
        ops.ElementWiseVectorizedUDF,
        ops.IntervalFromInteger,
        ops.IsInf,
        ops.IsNan,
        ops.Kurtosis,
        ops.Levenshtein,
        ops.MultiQuantile,
        ops.RandomUUID,
        ops.ReductionVectorizedUDF,
        ops.RegexExtract,
        ops.RegexReplace,
        ops.RegexSearch,
        ops.RegexSplit,
        ops.RowID,
        ops.Strftime,
        ops.StringJoin,
        ops.StringSplit,
        ops.StringToDate,
        ops.StringToTimestamp,
        ops.StringToTime,
        ops.TimeDelta,
        ops.TimestampAdd,
        ops.TimestampBucket,
        ops.TimestampDelta,
        ops.TimestampDiff,
        ops.TimestampSub,
        ops.TypeOf,
        ops.Unnest,
    )

    SIMPLE_OPS = {
        ops.Log10: "log10",
        ops.All: "min",
        ops.Any: "max",
    }

    @staticmethod
    def _minimize_spec(op, spec):
        if isinstance(func := op.func, ops.Analytic) and not isinstance(
            func, (ops.First, ops.Last, FirstValue, LastValue, ops.NthValue)
        ):
            return None
        return spec

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        """Exasol does not allow dots in quoted column names."""
        return name.replace(".", "_")

    def visit_Modulus(self, op, *, left, right):
        return self.f.anon.mod(left, right)

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

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                "Exasol only implements `pop` correlation coefficient"
            )

        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        if where is not None:
            arg = self.if_(where, arg, NULL)

        if order_by:
            arg = sge.Order(this=arg, expressions=order_by)

        return sge.GroupConcat(this=arg, separator=sep)

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the exasol backend"
            )
        return self.agg.first_value(arg, where=where, order_by=order_by)

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the exasol backend"
            )
        return self.agg.last_value(arg, where=where, order_by=order_by)

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
        return self.cast(self.f.floor(self.f.extract(self.v.second, arg)), op.dtype)

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.cast(
            (
                self.f.extract(self.v.second, arg)
                - self.f.floor(self.f.extract(self.v.second, arg))
            )
            * 1000,
            op.dtype,
        )

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.floor(self.f.posix_time(self.cast(arg, dt.timestamp)))

    def visit_StringConcat(self, op, *, arg):
        any_args_null = (a.is_(NULL) for a in arg)
        return self.if_(sg.or_(*any_args_null), NULL, self.f.concat(*arg))

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.cast(where, op.dtype))
        return self.f.count(STAR)

    def visit_CountDistinctStar(self, op, *, arg, where):
        cols = [sg.column(k, quoted=self.quoted) for k in op.arg.schema.keys()]
        if where is not None:
            cols = [self.if_(where, c, NULL) for c in cols]
        row = sge.Tuple(expressions=cols)
        return self.f.count(sge.Distinct(expressions=[row]))

    def visit_Median(self, op, *, arg, where):
        return self.visit_Quantile(op, arg=arg, quantile=sge.convert(0.5), where=where)

    visit_ApproxMedian = visit_Median

    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.WithinGroup(
            this=self.f[funcname](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )

    visit_ApproxQuantile = visit_Quantile

    def visit_TimestampTruncate(self, op, *, arg, unit):
        short_name = unit.short
        unit_mapping = {"W": "IW"}
        unsupported = {"ms", "us"}

        if short_name in unsupported:
            raise com.UnsupportedOperationError(
                f"Unsupported truncate unit {short_name}"
            )

        if short_name not in unit_mapping:
            return super().visit_TimestampTruncate(op, arg=arg, unit=unit)

        return self.f.date_trunc(unit_mapping[short_name], arg)

    def visit_DateTruncate(self, op, *, arg, unit):
        return self.visit_TimestampTruncate(op, arg=arg, unit=unit)

    def visit_DateDelta(self, op, *, part, left, right):
        # Note: general delta handling could be done based on part (unit),
        #       consider adapting this while implementing time based deltas.
        #       * part = day -> days_between
        #       * part = hour -> hours_between
        #       * ...
        return self.f.days_between(left, right)

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.cast(self.f.to_char(arg, "DDD"), op.dtype)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.cast(self.f.to_char(arg, "IW"), op.dtype)

    def visit_ExtractIsoYear(self, op, *, arg):
        return self.cast(self.f.to_char(arg, "IYYY"), op.dtype)

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.concat(
            self.f.substr(self.f.to_char(arg, "DAY"), 0, 1),
            self.f.trim(self.f.lower(self.f.substr(self.f.to_char(arg, "DAY"), 2))),
        )

    def visit_ExtractQuarter(self, op, *, arg):
        return self.cast(self.f.to_char(arg, "Q"), op.dtype)

    def visit_HexDigest(self, op, *, arg, how):
        ibis2exasol = {
            "md5": "hash_md5",
            "sha1": "hash_sha[1]",
            "sha256": "hash_sha256",
            "sha512": "hash_sha512",
        }
        how = how.lower()
        if how not in ibis2exasol:
            raise com.UnsupportedOperationError(
                f"Unsupported hashing algorithm ({how})"
            )
        func = self.f[ibis2exasol[how]]
        return func(arg)

    def visit_BitwiseLeftShift(self, op, *, left, right):
        return self.cast(self.f.bit_lshift(left, right), op.dtype)

    def visit_BitwiseRightShift(self, op, *, left, right):
        return self.cast(self.f.bit_rshift(left, right), op.dtype)

    def visit_BitwiseAnd(self, op, *, left, right):
        return self.cast(self.f.anon.bit_and(left, right), op.dtype)

    def visit_BitwiseOr(self, op, *, left, right):
        return self.cast(self.f.anon.bit_or(left, right), op.dtype)

    def visit_BitwiseXor(self, op, *, left, right):
        return self.cast(self.f.anon.bit_xor(left, right), op.dtype)


compiler = ExasolCompiler()
