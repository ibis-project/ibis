"""Db2 SQL compiler for Ibis expressions."""

from __future__ import annotations

import string

import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR, SQLGlotCompiler
from ibis.backends.sql.datatypes import Db2Type
from ibis.backends.sql.dialects import Db2
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    lower_log2,
    lower_log10,
    lower_sample,
    split_select_distinct_with_order_by,
)


class Db2Compiler(SQLGlotCompiler):
    """SQL compiler for the Db2 backend."""

    __slots__ = ()

    dialect = Db2
    type_mapper = Db2Type

    LOWERED_OPS = {
        ops.Capitalize: None,
        ops.Log2: lower_log2,
        ops.Log10: lower_log10,
    }

    rewrites = (
        exclude_unsupported_window_frame_from_ops | lower_sample(),
        *SQLGlotCompiler.rewrites,
    )

    post_rewrites = (split_select_distinct_with_order_by,)

    UNSUPPORTED_OPS = (
        # aggregation
        ops.ArgMin,
        ops.ArgMax,
        ops.Kurtosis,
        ops.Mode,
        ops.First,
        ops.Last,
        ops.CountDistinctStar,
        # string
        ops.Reverse,
        ops.StringSplit,
        ops.StringToDate,
        ops.StringToTimestamp,
        ops.Levenshtein,
        ops.RegexSplit,
        # date/time
        ops.DateFromYMD,
        ops.TimeDelta,
        ops.ExtractEpochSeconds,
        ops.Strftime,
        ops.TimestampBucket,
        # numeric
        ops.IsNan,
        ops.IsInf,
        # JSON
        ops.JSONGetItem,
        # logical
        ops.Xor,
        ops.BitwiseLeftShift,
        ops.BitwiseRightShift,
        # array
        ops.Array,
        ops.ArrayCollect,
        ops.ArrayConcat,
        ops.ArrayContains,
        ops.ArrayFilter,
        ops.ArrayFlatten,
        ops.ArrayIndex,
        ops.ArrayLength,
        ops.ArrayMap,
        ops.ArrayRemove,
        ops.ArrayRepeat,
        ops.ArraySort,
        ops.ArrayStringJoin,
        ops.ArrayUnion,
        ops.ArrayZip,
        ops.Unnest,
        # map
        ops.Map,
        ops.MapGet,
        ops.MapKeys,
        ops.MapMerge,
        ops.MapValues,
        # struct
        ops.StructColumn,
        ops.StructField,
        # hash
        ops.Hash,
        ops.HashBytes,
    )

    # Keep this empty; do not delete it. `__init_subclass__` runs after the class
    # body and applies every entry of the inherited SIMPLE_OPS table, which
    # overwrites `visit_*` methods the class already defines (for example
    # StringLength, Atan2, Translate and Any).
    SIMPLE_OPS = {}

    @staticmethod
    def _generate_groups(groups):
        """Generate GROUP BY clause."""
        return groups

    @staticmethod
    def _minimize_spec(op, spec):
        # Db2 accepts a window frame only on aggregates and on
        # FIRST_VALUE/LAST_VALUE/NTH_VALUE; ranking and shift functions
        # (ROW_NUMBER, RANK, NTILE, LAG, ...) raise a syntax error with one.
        if isinstance(func := op.func, ops.Analytic) and not isinstance(
            func, (ops.First, ops.Last, FirstValue, LastValue, ops.NthValue)
        ):
            return None
        return spec

    def visit_WindowFunction(self, op, *, how, func, start, end, group_by, order_by):
        """Visit a WindowFunction operation."""
        name = type(op.func).__name__
        if isinstance(
            op.func,
            (ops.CountDistinct, ops.ApproxCountDistinct, ops.Median, ops.ApproxMedian),
        ):
            # Db2 rejects these aggregates inside OVER (SQL0441N, SQL0104N).
            raise com.OperationNotDefinedError(
                f"Db2 does not support {name} over a window"
            )
        if isinstance(op.func, (ops.Any, ops.All)) and all(
            isinstance(key.arg, ops.Literal) and key.arg.value is None
            for key in op.order_by
        ):
            # The shared window rewrite gives an unordered Any/All the sort key
            # `ORDER BY NULL`, which Db2 rejects with an aggregate (SQL0418N).
            raise com.OperationNotDefinedError(
                f"Db2 does not support {name} over a window without an order_by"
            )
        window = super().visit_WindowFunction(
            op,
            how=how,
            func=func,
            start=start,
            end=end,
            group_by=group_by,
            order_by=order_by,
        )
        if isinstance(op.func, (ops.CumeDist, ops.PercentRank)):
            # Db2 returns a type the ibm_db driver cannot fetch (SystemError);
            # casting the whole windowed expression to DOUBLE makes it readable.
            return self.cast(window, dt.float64)
        return window

    def visit_Pi(self, op):
        # Db2 11.5 has no PI(); ACOS(-1) works on 11.5 and 12.1.
        return self.f.acos(-1)

    def visit_Cast(self, op, *, arg, to, **kwargs):
        from_ = op.arg.dtype
        if to.is_interval():
            return self._make_interval(arg, to.unit)
        if to.is_timestamp() and (from_.is_integer() or from_.is_floating()):
            # Db2 cannot CAST a number to TIMESTAMP; add the seconds to the epoch.
            seconds = self.f.floor(arg) if from_.is_floating() else arg
            result = self.f.timestamp(
                sge.convert("1970-01-01 00:00:00")
            ) + sge.Interval(this=self.cast(seconds, dt.int64), unit=self.v.SECOND)
            if from_.is_floating():
                micros = self.cast((arg - self.f.floor(arg)) * 1000000, dt.int64)
                result += sge.Interval(this=micros, unit=self.v.MICROSECOND)
            return self.cast(result, to)
        if op.arg.dtype.is_boolean() and to.is_string():
            # CAST(boolean AS VARCHAR) renders TRUE/FALSE; use the lowercase
            # rendering that the other backends produce. NULL matches neither
            # branch, so it stays NULL.
            return sge.Case(
                ifs=[
                    sge.If(this=arg, true=sge.Literal.string("true")),
                    sge.If(this=sge.Not(this=arg), true=sge.Literal.string("false")),
                ]
            )
        if op.arg.dtype.is_string() and (
            to.is_date() or to.is_time() or to.is_timestamp()
        ):
            # Db2 11.5 types a long string concatenation as LONG VARCHAR, which
            # it cannot cast to a datetime type. Casting to VARCHAR(32672)
            # first makes it castable; Db2 12.1 accepts both forms.
            arg = self.cast(arg, dt.string)
        return self.cast(arg, to)

    def visit_Log(self, op, *, arg, base):
        # Db2 has LN and LOG10 but no two-argument LOG(base, x); other bases
        # use LN(x) / LN(base).
        if base is None:
            return self.f.ln(arg)
        base_value = op.base.value if isinstance(op.base, ops.Literal) else None
        if base_value == 10:
            # self.f.log10 produces LOG(10, x) on some sqlglot versions.
            return self.f.anon.log10(arg)
        if base_value == 2:
            return self.f.ln(arg) / self.f.ln(sge.convert(2))
        return self.f.ln(arg) / self.f.ln(base)

    def visit_StringContains(self, op, *, haystack, needle, **kwargs):
        return self.f.anon.locate(needle, haystack) > 0

    def visit_EndsWith(self, op, *, arg, end, **kwargs):
        # RIGHT/LENGTH treat the value literally, so % and _ need no escaping.
        return sge.Right(this=arg, expression=sge.Length(this=end)).eq(end)

    def visit_StartsWith(self, op, *, arg, start, **kwargs):
        # LEFT/LENGTH treat the value literally, so % and _ need no escaping.
        return sge.Left(this=arg, expression=sge.Length(this=start)).eq(start)

    @staticmethod
    def _char_units():
        # Db2 string functions count bytes when STRING_UNITS is SYSTEM (the
        # default), so non-ASCII text gives wrong lengths and positions.
        # CODEUNITS32 makes them count characters.
        return sge.Var(this="CODEUNITS32")

    def visit_StringLength(self, op, *, arg):
        return self.f.anon.character_length(arg, self._char_units())

    def visit_Substring(self, op, *, arg, start, length):
        if isinstance(op.length, ops.Literal) and (value := op.length.value) < 0:
            raise com.IbisInputError(
                f"Length parameter must be a non-negative value; got {value}"
            )
        units = self._char_units()
        start += 1
        start = self.if_(
            start >= 1, start, start + self.f.anon.character_length(arg, units)
        )
        if length is None:
            return self.f.anon.substring(arg, start, units)
        return self.f.anon.substring(arg, start, length, units)

    def visit_StrRight(self, op, *, arg, nchars):
        # RIGHT pads with blanks when nchars exceeds the string length.
        units = self._char_units()
        nchars = self.f.least(nchars, self.f.anon.character_length(arg, units))
        return self.f.anon.right(arg, nchars, units)

    def visit_Atan2(self, op, *, left, right):
        # Db2 takes the x coordinate first: ATAN2(x, y).
        return self.f.atan2(right, left)

    def visit_Translate(self, op, *, arg, from_str, to_str):
        # Db2 takes the replacement characters before the ones to replace:
        # TRANSLATE(string, to, from).
        return self.f.translate(arg, to_str, from_str)

    def visit_StringFind(self, op, *, arg, substr, start, end, **kwargs):
        # LOCATE is 1-based, ibis `start` is 0-based; the 1->0 conversion of the
        # result is done by the `one_to_zero_index` rewrite, so only `start`
        # needs shifting here.
        start = sge.convert(1) if start is None else start + sge.convert(1)
        return self.f.anon.locate(substr, arg, start, self._char_units())

    def visit_RegexExtract(self, op, *, arg, pattern, index, **kwargs):
        # REGEXP_SUBSTR(source, pattern, start, occurrence, flags, group)
        return self.f.anon.regexp_substr(arg, pattern, 1, 1, "", index)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement, **kwargs):
        return self.f.anon.regexp_replace(arg, pattern, replacement)

    def visit_Arbitrary(self, op, *, arg, where):
        # Db2 has no ANY_VALUE. Arbitrary only promises *some* non-null value, so MIN is a valid
        # choice: it ignores NULLs and is deterministic. `where` is folded
        # into a CASE by `self.agg` because Db2 has no FILTER clause.
        return self.agg.min(arg, where=where)

    def visit_ApproxMedian(self, op, *, arg, where):
        # Db2 has no APPROX_QUANTILE.
        return self.visit_Median(op, arg=arg, where=where)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, None)
        return sge.Count(this=sge.Distinct(expressions=[arg]))

    def visit_ApproxCountDistinct(self, op, *, arg, where):
        # Db2 has no APPROX_COUNT_DISTINCT.
        return self.visit_CountDistinct(op, arg=arg, where=where)

    @staticmethod
    def _bool_as_int(arg):
        # Db2 has no boolean aggregates (BOOL_OR, or MAX over a boolean), so
        # aggregate the argument as 1/0. A NULL argument stays NULL, because
        # neither WHEN branch fires for it and there is no ELSE; MAX and MIN
        # then ignore it, as Ibis expects.
        return sge.Case(
            ifs=[
                sge.If(this=arg, true=sge.convert(1)),
                sge.If(this=sge.Not(this=arg), true=sge.convert(0)),
            ]
        )

    def visit_Any(self, op, *, arg, where):
        arg = self._bool_as_int(arg)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Max(this=arg)

    def visit_All(self, op, *, arg, where):
        arg = self._bool_as_int(arg)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Min(this=arg)

    def visit_Mean(self, op, *, arg, where):
        dtype = op.arg.dtype
        if dtype.is_boolean():
            # Db2 cannot cast BOOLEAN directly to DOUBLE (SQL0461N); go through
            # int32 first.
            arg = self.cast(arg, dt.int32)
        if dtype.is_boolean() or dtype.is_integer():
            # AVG over an integer column returns an integer in Db2, which
            # truncates the mean.
            arg = self.cast(arg, dt.float64)
        return self.agg.avg(arg, where=where)

    def visit_CountStar(self, op, *, arg, where):
        # Db2 does not accept `*` inside a CASE expression, so count a
        # constant for the rows that match `where` instead.
        if where is not None:
            return self.f.count(self.if_(where, 1, NULL))
        return self.f.count(STAR)

    def visit_GroupConcat(self, op, *, arg, where, sep, order_by):
        if order_by:
            # LISTAGG ... WITHIN GROUP rejects the NULLS FIRST/LAST that Ibis
            # emits (SQL0109N); dropping it would change the null ordering.
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                f"not supported for the {self.dialect} backend"
            )
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.listagg(arg, sep)

    def visit_Capitalize(self, op, *, arg):
        # The default lowering emits SUBSTR(s, 2, LENGTH(s)). Db2 pads that
        # result with blanks on VARCHAR columns, raises SQL0138N on literals,
        # and counts bytes rather than characters. SUBSTRING with CODEUNITS32
        # counts characters, and omitting the length returns the rest.
        units = self._char_units()
        first = self.f.anon.substring(arg, 1, 1, units)
        rest = self.f.anon.substring(arg, 2, units)
        return self.if_(
            self.f.anon.character_length(arg, units) < 2,
            self.f.upper(arg),
            sge.DPipe(this=self.f.upper(first), expression=self.f.lower(rest)),
        )

    # Db2's TRIM removes a single character; LTRIM/RTRIM take a set. `f.anon`
    # keeps sqlglot from parsing them into `exp.Trim` (rendered as TRIM).
    def visit_Strip(self, op, *, arg):
        return self.f.anon.rtrim(
            self.f.anon.ltrim(arg, string.whitespace), string.whitespace
        )

    def visit_LStrip(self, op, *, arg):
        return self.f.anon.ltrim(arg, string.whitespace)

    def visit_RStrip(self, op, *, arg):
        return self.f.anon.rtrim(arg, string.whitespace)

    def visit_StringJoin(self, op, *, arg, sep):
        # Db2 has no CONCAT_WS. `||` returns NULL if any operand is NULL, but
        # CONCAT_WS skips NULL operands, so prefix every non-NULL operand with
        # the separator and drop the leading one. All-NULL input gives '' and a
        # NULL separator gives NULL, as with CONCAT_WS.
        terms = [
            self.if_(a.is_(NULL), sge.convert(""), self.f.concat(sep, a)) for a in arg
        ]
        return self.f.substr(self.f.concat(*terms), self.f.length(sep) + 1)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.f.date(value.isoformat())
        if dtype.is_timestamp() and value.utcoffset():
            # Db2 timestamps have no time zone and reject an offset in the
            # string, so a non-zero offset can't be represented. `utcoffset()`
            # is None for a naive value and a zero timedelta for UTC.
            raise com.UnsupportedOperationError(
                "Db2 timestamps have no time zone; cannot compile a timestamp "
                f"literal with a non-zero UTC offset: {value}"
            )
        if dtype.is_timestamp() and value.tzinfo is not None:
            # A zero offset is stored as a naive UTC value.
            return self.cast(
                value.replace(tzinfo=None).isoformat(" "), dtype.copy(timezone=None)
            )
        if dtype.is_interval():
            return self._make_interval(sge.convert(value), dtype.unit)
        return None  # fall back to the default literal handling

    def _make_interval(self, arg, unit):
        # Db2 labeled durations have no WEEK, QUARTER or MILLISECOND label.
        factor, label = {
            "year": (1, "YEAR"),
            "quarter": (3, "MONTH"),
            "month": (1, "MONTH"),
            "week": (7, "DAY"),
            "day": (1, "DAY"),
            "hour": (1, "HOUR"),
            "minute": (1, "MINUTE"),
            "second": (1, "SECOND"),
            "millisecond": (1000, "MICROSECOND"),
            "microsecond": (1, "MICROSECOND"),
        }.get(unit.singular, (None, None))
        if label is None:
            raise com.UnsupportedOperationError(
                f"Db2 has no {unit.singular} labeled duration"
            )
        if factor != 1:
            arg = arg * factor
        return sge.Interval(this=arg, unit=self.v[label])

    # `a - b` on Db2 dates, timestamps and times is a packed-decimal duration
    # (YYYYMMDD...), not a count of days or seconds.
    def visit_DateDiff(self, op, *, left, right):
        return self.f.days_between(left, right)

    def visit_TimestampDiff(self, op, *, left, right):
        # SECONDS_BETWEEN truncates toward zero; floor the exact difference so
        # negative fractions round the same way as the other backends.
        days = self.cast(self.f.days(left), dt.int64) - self.f.days(right)
        secs = self.f.midnight_seconds(left) - self.f.midnight_seconds(right)
        frac = (
            self.f.microsecond(left) - self.f.microsecond(right)
        ) / sge.Literal.number("1000000.0")
        return days * 86400 + self.cast(self.f.floor(secs + frac), dt.int64)

    def visit_TimeDiff(self, op, *, left, right):
        return self.f.midnight_seconds(left) - self.f.midnight_seconds(right)

    def visit_TimestampDelta(self, op, *, part, left, right):
        # Count the unit boundaries crossed: truncate both sides to the unit,
        # then take the exact difference. `*_BETWEEN` alone would count whole
        # elapsed units instead.
        if not isinstance(part, sge.Literal):
            raise com.UnsupportedOperationError(
                "Only literal `part` values are supported for delta"
            )
        between, fmt = {
            "year": ("YEARS_BETWEEN", "YEAR"),
            "month": ("MONTHS_BETWEEN", "MM"),
            "week": ("WEEKS_BETWEEN", "IW"),
            "day": ("DAYS_BETWEEN", "DD"),
            "hour": ("HOURS_BETWEEN", "HH"),
            "minute": ("MINUTES_BETWEEN", "MI"),
            "second": ("SECONDS_BETWEEN", "SS"),
        }.get(part.this, (None, None))
        if between is None or (
            isinstance(op, ops.DateDelta) and fmt in ("HH", "MI", "SS")
        ):
            raise com.UnsupportedOperationError(
                f"Unsupported delta part {part.this!r} for {op.left.dtype}"
            )
        return self.cast(
            self.f[between](self.f.trunc(left, fmt), self.f.trunc(right, fmt)),
            dt.int64,
        )

    visit_DateDelta = visit_TimestampDelta

    def visit_DateTruncate(self, op, *, arg, unit):
        """Visit a DateTruncate operation."""
        trunc_unit_mapping = {
            "Y": "YEAR",
            "Q": "Q",
            "M": "MM",
            "W": "IW",  # ISO week, starts on Monday ('W' and 'DAY' do not)
            "D": "DD",  # 'DAY' and 'D' truncate to the start of the week
            "h": "HH",
            "m": "MI",
            "s": "SS",
        }
        subsecond_mapping = {"ms": "FF3", "us": "FF6"}

        if (fmt := subsecond_mapping.get(unit.short)) is not None:
            # TRUNC cannot truncate below a second; round-trip through a format.
            pattern = sge.Literal.string(f"YYYY-MM-DD HH24:MI:SS.{fmt}")
            return self.f.anon.timestamp_format(
                self.f.anon.varchar_format(arg, pattern), pattern
            )

        if (mapped := trunc_unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")
        return self.f.anon.trunc(arg, mapped)

    def visit_TimestampTruncate(self, op, *, arg, unit):
        return self.visit_DateTruncate(op, arg=arg, unit=unit)

    def visit_ExtractSecond(self, op, *, arg):
        # EXTRACT(SECOND ...) returns the fractional seconds; SECOND() does not.
        return self.f.anon.second(arg)

    def visit_DayOfWeekIndex(self, op, *, arg):
        # DAYOFWEEK is 1=Sunday..7=Saturday; Ibis wants 0=Monday..6=Sunday.
        return sge.Mod(this=self.f.anon.dayofweek(arg) + 5, expression=sge.convert(7))

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.anon.dayname(arg)

    def visit_RandomScalar(self, op):
        return self.f.anon.rand()

    def visit_RandomUUID(self, op):
        # Built-in since Db2 12.1: returns a version 4 UUID as CHAR(36), in
        # uppercase; Ibis (like Python's uuid) uses lowercase.
        # (GENERATE_UNIQUE() is not a UUID.)
        return self.f.lower(self.f.anon.generate_uuid())


compiler = Db2Compiler()
