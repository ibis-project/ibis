"""Db2 SQL compiler for Ibis expressions."""

from __future__ import annotations

import sqlglot.expressions as sge
from db2_sqlglot import Db2 as _Db2BaseDialect
from sqlglot import exp, transforms
from sqlglot.generator import Generator

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR, SQLGlotCompiler
from ibis.backends.sql.datatypes import Db2Type
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    lower_log2,
    lower_log10,
    lower_sample,
    split_select_distinct_with_order_by,
)


class Db2Generator(Generator):
    """Custom SQL generator for Db2."""

    # Only Ibis-specific keys — no base spread. The base Generator keys are
    # inherited by Db2Dialect.Generator via the MRO and must not be spread here,
    # because spreading **Generator.TRANSFORMS would overwrite the external
    # dialect's specialised transforms for keys that appear in both dicts.
    TRANSFORMS = {
        exp.DateDiff: lambda self, e: self.func(
            "DAYS_BETWEEN", e.this, self.sql(e, "expression")
        ),
        exp.StrPosition: lambda self, e: self.func(
            "LOCATE", self.sql(e, "substr"), self.sql(e, "this")
        ),
        exp.GroupConcat: lambda self, e: self.func(
            "LISTAGG",
            e.this,
            self.sql(e, "separator")
            if e.args.get("separator")
            else self.sql(exp.Literal.string(",")),
        ),
        # Db2's STDDEV is the population standard deviation; STDDEV_SAMP is
        # the sample one that other engines mean by STDDEV.
        exp.Stddev: lambda self, e: self.func("STDDEV_SAMP", e.this),
        # Likewise, Db2's VARIANCE is the population variance; VAR_SAMP and
        # VAR_POP are the explicit sample and population forms.
        exp.Variance: lambda self, e: self.func("VAR_SAMP", e.this),
        exp.VariancePop: lambda self, e: self.func("VAR_POP", e.this),
    }

    # Db2 rejects the RECURSIVE keyword (SQL0104N) and requires explicit
    # column names on a recursive common table expression (SQL0343N).
    CTE_RECURSIVE_KEYWORD_REQUIRED = False

    def cte_sql(self, expression: exp.CTE) -> str:
        alias = expression.args.get("alias")
        with_ = expression.parent
        if (
            alias is not None
            and not alias.args.get("columns")
            and with_ is not None
            and with_.args.get("recursive")
        ):
            alias.set(
                "columns",
                [
                    exp.to_identifier(name, quoted=True)
                    for name in expression.this.named_selects
                ],
            )
        return super().cte_sql(expression)

    # Only Ibis-specific overrides — no **Generator.TYPE_MAPPING spread.
    TYPE_MAPPING = {
        exp.DataType.Type.BOOLEAN: "BOOLEAN",
        exp.DataType.Type.TINYINT: "SMALLINT",
        exp.DataType.Type.SMALLINT: "SMALLINT",
        exp.DataType.Type.INT: "INTEGER",
        exp.DataType.Type.BIGINT: "BIGINT",
        exp.DataType.Type.FLOAT: "REAL",
        exp.DataType.Type.DOUBLE: "DOUBLE",
        exp.DataType.Type.DECIMAL: "DECIMAL",
        exp.DataType.Type.VARCHAR: "VARCHAR",
        exp.DataType.Type.CHAR: "CHAR",
        exp.DataType.Type.TEXT: "CLOB",
        exp.DataType.Type.BINARY: "VARBINARY",
        exp.DataType.Type.VARBINARY: "VARBINARY",
        exp.DataType.Type.DATE: "DATE",
        exp.DataType.Type.DATETIME: "TIMESTAMP",
        exp.DataType.Type.TIMESTAMP: "TIMESTAMP",
        exp.DataType.Type.TIMESTAMPTZ: "TIMESTAMP",
    }

    def interval_sql(self, expression: exp.Interval) -> str:
        """Render a Db2 labeled duration: `(n) DAYS`, not `INTERVAL n DAY`."""
        return f"({self.sql(expression, 'this')}) {self.sql(expression, 'unit')}"

    def cast_sql(self, expression: exp.Cast, safe_prefix: str | None = None) -> str:
        """Generate CAST expression for Db2."""
        return f"CAST({self.sql(expression, 'this')} AS {self.sql(expression, 'to')})"

    def trycast_sql(
        self, expression: exp.TryCast, safe_prefix: str | None = None
    ) -> str:
        """Db2 doesn't have TRY_CAST, use regular CAST."""
        return self.cast_sql(expression, safe_prefix)

    def boolean_sql(self, expression: exp.Boolean) -> str:
        """Generate boolean literal for Db2."""
        return "TRUE" if expression.this else "FALSE"

    def concat_sql(self, expression: exp.Concat) -> str:
        """Generate CONCAT expression for Db2.

        Db2's CONCAT() takes exactly two arguments. Use || for arbitrary
        numbers of operands, which is valid for any count. Parenthesise
        to preserve precedence inside larger expressions.
        """
        return "(" + " || ".join(self.sql(e) for e in expression.expressions) + ")"

    def substring_sql(self, expression: exp.Substring) -> str:
        """Generate SUBSTRING expression for Db2."""
        # Db2 uses SUBSTR function
        args = [expression.this]
        if expression.args.get("start"):
            args.append(expression.args["start"])
        if expression.args.get("length"):
            args.append(expression.args["length"])
        return self.func("SUBSTR", *args)

    def regexp_like_sql(self, expression: exp.RegexpLike) -> str:
        """Generate REGEXP_LIKE for Db2."""
        return self.func("REGEXP_LIKE", expression.this, expression.expression)

    def mod_sql(self, expression: exp.Mod) -> str:
        """Generate modulo for Db2."""
        return self.func("MOD", expression.this, expression.expression)

    def sqrt_sql(self, expression: exp.Sqrt) -> str:
        """Generate square root for Db2."""
        return self.func("SQRT", expression.this)

    def power_sql(self, expression: exp.Pow) -> str:
        """Generate power function for Db2."""
        return self.func("POWER", expression.this, expression.expression)


class Db2Dialect(_Db2BaseDialect):
    """Db2 SQL dialect for SQLGlot, extending the base dialect from db2_sqlglot."""

    class Generator(Db2Generator, _Db2BaseDialect.Generator):
        """Extended Db2 generator with Ibis-specific customisations.

        Db2Generator comes first in the MRO so all Ibis-defined methods
        (boolean_sql, cast_sql, trycast_sql, extract_sql, etc.) take
        precedence over the external dialect's defaults.  The external
        generator fills in anything not overridden here.
        """

        TYPE_MAPPING = {
            **_Db2BaseDialect.Generator.TYPE_MAPPING,
            **Db2Generator.TYPE_MAPPING,
        }

        TRANSFORMS = {
            # External dialect is the base; Db2Generator keys are the only
            # overrides, so they win for every key they define.
            **_Db2BaseDialect.Generator.TRANSFORMS,
            **Db2Generator.TRANSFORMS,
            # Db2 has no SEMI/ANTI JOIN syntax; rewrite them as [NOT] EXISTS
            # before the external dialect's own Select transform runs.
            exp.Select: lambda self, e: _Db2BaseDialect.Generator.TRANSFORMS[
                exp.Select
            ](self, transforms.eliminate_semi_and_anti_joins(e)),
        }

    class Parser(_Db2BaseDialect.Parser):
        """Extended Db2 parser with Ibis-specific customisations."""

        FUNCTIONS = {
            **_Db2BaseDialect.Parser.FUNCTIONS,
            "LOCATE": exp.StrPosition.from_arg_list,
            "LISTAGG": exp.GroupConcat.from_arg_list,
            "DAYS_BETWEEN": exp.DateDiff.from_arg_list,
            "REGEXP_LIKE": lambda args: exp.RegexpLike(
                this=args[0], expression=args[1]
            ),
        }


class Db2Compiler(SQLGlotCompiler):
    """SQL compiler for the Db2 backend."""

    __slots__ = ()

    dialect = Db2Dialect
    type_mapper = Db2Type

    LOWERED_OPS = {
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
        ops.Mode,
        ops.First,
        ops.Last,
        ops.CountDistinctStar,
        # string
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
        # numeric
        ops.IsNan,
        ops.IsInf,
        # JSON
        ops.JSONGetItem,
        # logical
        ops.Xor,
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

    # Exclude operations from SIMPLE_OPS that need custom implementations
    SIMPLE_OPS = {
        k: v
        for k, v in SQLGlotCompiler.SIMPLE_OPS.items()
        if k
        not in (
            ops.StartsWith,  # → visit_StartsWith (LEFT + LENGTH)
            ops.Any,  # → visit_Any (MAX over CASE)
            ops.All,  # → visit_All (MIN over CASE)
            ops.StringContains,  # → visit_StringContains (LOCATE)
            ops.ApproxCountDistinct,  # → visit_ApproxCountDistinct (COUNT DISTINCT)
            ops.RandomUUID,  # → visit_RandomUUID (GENERATE_UUID)
            ops.Median,  # → visit_Median (MEDIAN)
            ops.RegexSearch,  # → visit_RegexSearch (REGEXP_LIKE)
            ops.RegexExtract,  # → visit_RegexExtract (REGEXP_SUBSTR)
            ops.RandomScalar,  # → visit_RandomScalar (RAND())
        )
    }

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

    def visit_Cast(self, op, *, arg, to, **kwargs):
        """Visit a Cast operation."""
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
        # Db2 uses CAST syntax
        return self.cast(arg, to)

    def visit_TryCast(self, op, *, arg, to, **kwargs):
        """Visit a TryCast operation."""
        # Db2 has no TRY_CAST; fall back to regular CAST.
        return self.cast(arg, to)

    def visit_Log(self, op, *, arg, base):
        """Logarithm for Db2.

        Db2 has LN (natural log) and LOG10 (base 10), but no two-argument
        LOG(base, x).  Arbitrary bases use LN(x) / LN(base).
        Log2 and Log10 are lowered by LOWERED_OPS to ops.Log(arg, base=2/10)
        before this method runs.
        """
        if base is None:
            return self.f.ln(arg)
        base_value = op.base.value if isinstance(op.base, ops.Literal) else None
        if base_value == 10:
            # self.f.log10 produces LOG(10, x) on some sqlglot versions;
            # use Anonymous to guarantee LOG10(x).
            return sge.Anonymous(this="LOG10", expressions=[arg])
        if base_value == 2:
            return self.f.ln(arg) / self.f.ln(sge.convert(2))
        return self.f.ln(arg) / self.f.ln(base)

    def visit_StringContains(self, op, *, haystack, needle, **kwargs):
        """Visit a StringContains operation."""
        # Db2 uses LOCATE function
        return sge.GT(
            this=sge.Anonymous(this="LOCATE", expressions=[needle, haystack]),
            expression=sge.convert(0),
        )

    def visit_EndsWith(self, op, *, arg, end, **kwargs):
        """Visit an EndsWith operation.

        Generates SQL: RIGHT(arg, LENGTH(end)) = end
        Db2 built-ins RIGHT() and LENGTH() treat the value as a literal,
        so wildcard characters (% and _) require no escaping.
        """
        return sge.Right(this=arg, expression=sge.Length(this=end)).eq(end)

    def visit_StartsWith(self, op, *, arg, start, **kwargs):
        """Visit a StartsWith operation.

        Generates SQL: LEFT(arg, LENGTH(start)) = start
        Db2 built-ins LEFT() and LENGTH() treat the value as a literal,
        so wildcard characters (% and _) require no escaping.
        """
        return sge.Left(this=arg, expression=sge.Length(this=start)).eq(start)

    def visit_StringFind(self, op, *, arg, substr, start, end, **kwargs):
        """Visit a StringFind operation."""
        # Db2 uses LOCATE function. LOCATE is 1-based, ibis `start` is 0-based;
        # the 1->0 conversion of the result is done by the `one_to_zero_index`
        # rewrite, so only `start` needs shifting here.
        if start is not None:
            return sge.Anonymous(
                this="LOCATE", expressions=[substr, arg, start + sge.convert(1)]
            )
        return sge.Anonymous(this="LOCATE", expressions=[substr, arg])

    def visit_RegexSearch(self, op, *, arg, pattern, **kwargs):
        """Visit a RegexSearch operation."""
        # Db2 uses REGEXP_LIKE function
        return sge.Anonymous(this="REGEXP_LIKE", expressions=[arg, pattern])

    def visit_RegexExtract(self, op, *, arg, pattern, index, **kwargs):
        """Visit a RegexExtract operation."""
        # Db2 uses REGEXP_SUBSTR(source, pattern, start, occurrence, flags, group)
        return sge.Anonymous(
            this="REGEXP_SUBSTR",
            expressions=[
                arg,
                pattern,
                sge.convert(1),
                sge.convert(1),
                sge.convert(""),
                index,
            ],
        )

    def visit_RegexReplace(self, op, *, arg, pattern, replacement, **kwargs):
        """Visit a RegexReplace operation."""
        # Db2 uses REGEXP_REPLACE function
        return sge.Anonymous(
            this="REGEXP_REPLACE", expressions=[arg, pattern, replacement]
        )

    def visit_Arbitrary(self, op, *, arg, where):
        """Visit an Arbitrary operation using MIN (Db2 does not have ANY_VALUE)."""
        # Arbitrary only promises *some* non-null value, so MIN is a valid
        # choice: it ignores NULLs and is deterministic. `where` is folded
        # into a CASE by `self.agg` because Db2 has no FILTER clause.
        return self.agg.min(arg, where=where)

    def visit_Median(self, op, *, arg, where):
        """Visit a Median operation."""
        # Db2 uses MEDIAN function; fold where into arg as CASE (no FILTER support)
        if where is not None:
            arg = self.if_(where, arg, None)
        return sge.Anonymous(this="MEDIAN", expressions=[arg])

    def visit_ApproxMedian(self, op, *, arg, where):
        """Visit an ApproxMedian operation using MEDIAN (Db2 has no APPROX_QUANTILE)."""
        return self.visit_Median(op, arg=arg, where=where)

    def visit_CountDistinct(self, op, *, arg, where):
        """Visit a CountDistinct operation."""
        if where is not None:
            arg = self.if_(where, arg, None)
        return sge.Count(this=sge.Distinct(expressions=[arg]))

    def visit_ApproxCountDistinct(self, op, *, arg, where):
        """Visit an ApproxCountDistinct operation."""
        # Db2 doesn't have APPROX_COUNT_DISTINCT, use COUNT(DISTINCT)
        return self.visit_CountDistinct(op, arg=arg, where=where)

    def visit_Any(self, op, *, arg, where):
        # Db2 has no boolean aggregates (BOOL_OR, or MAX over a boolean), so
        # aggregate the argument as 1/0.
        arg = self.if_(arg, 1, 0)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Max(this=arg)

    def visit_All(self, op, *, arg, where):
        arg = self.if_(arg, 1, 0)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Min(this=arg)

    def visit_CountStar(self, op, *, arg, where):
        # Db2 does not accept `*` inside a CASE expression, so count a
        # constant for the rows that match `where` instead.
        if where is not None:
            return self.f.count(self.if_(where, 1, NULL))
        return self.f.count(STAR)

    def visit_Lag(self, op, *, arg, offset, default):
        """Visit a Lag operation."""
        # Db2 LAG function
        expressions = [arg]
        if offset is not None:
            expressions.append(offset)
        if default is not None:
            expressions.append(default)
        return sge.Anonymous(this="LAG", expressions=expressions)

    def visit_Lead(self, op, *, arg, offset, default):
        """Visit a Lead operation."""
        # Db2 LEAD function
        expressions = [arg]
        if offset is not None:
            expressions.append(offset)
        if default is not None:
            expressions.append(default)
        return sge.Anonymous(this="LEAD", expressions=expressions)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.f.date(value.isoformat())
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
        return self.f.seconds_between(left, right)

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
            varchar_format = sge.Anonymous(
                this="VARCHAR_FORMAT", expressions=[arg, pattern]
            )
            return sge.Anonymous(
                this="TIMESTAMP_FORMAT", expressions=[varchar_format, pattern]
            )

        if (mapped := trunc_unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")
        return sge.Anonymous(this="TRUNC", expressions=[arg, sge.convert(mapped)])

    def visit_TimestampTruncate(self, op, *, arg, unit):
        """Visit a TimestampTruncate operation."""
        # Similar to DateTruncate but for timestamps
        return self.visit_DateTruncate(op, arg=arg, unit=unit)

    def visit_ExtractYear(self, op, *, arg):
        """Visit an ExtractYear operation."""
        return sge.Anonymous(this="YEAR", expressions=[arg])

    def visit_ExtractMonth(self, op, *, arg):
        """Visit an ExtractMonth operation."""
        return sge.Anonymous(this="MONTH", expressions=[arg])

    def visit_ExtractDay(self, op, *, arg):
        """Visit an ExtractDay operation."""
        return sge.Anonymous(this="DAY", expressions=[arg])

    def visit_ExtractHour(self, op, *, arg):
        """Visit an ExtractHour operation."""
        return sge.Anonymous(this="HOUR", expressions=[arg])

    def visit_ExtractMinute(self, op, *, arg):
        """Visit an ExtractMinute operation."""
        return sge.Anonymous(this="MINUTE", expressions=[arg])

    def visit_ExtractSecond(self, op, *, arg):
        """Visit an ExtractSecond operation."""
        return sge.Anonymous(this="SECOND", expressions=[arg])

    def visit_DayOfWeekIndex(self, op, *, arg):
        """Visit a DayOfWeekIndex operation."""
        # Db2 uses DAYOFWEEK function (1=Sunday, 7=Saturday)
        # ibis expects 0=Monday, 1=Tuesday, ..., 6=Sunday
        # Formula: (DAYOFWEEK(x) + 5) % 7
        return sge.Mod(
            this=sge.Add(
                this=sge.Anonymous(this="DAYOFWEEK", expressions=[arg]),
                expression=sge.convert(5),
            ),
            expression=sge.convert(7),
        )

    def visit_DayOfWeekName(self, op, *, arg):
        """Visit a DayOfWeekName operation."""
        # Db2 uses DAYNAME function
        return sge.Anonymous(this="DAYNAME", expressions=[arg])

    def visit_RandomScalar(self, op):
        """Visit a RandomScalar operation."""
        # Db2 uses RAND() function
        return sge.Anonymous(this="RAND", expressions=[])

    def visit_RandomUUID(self, op):
        # Built-in since Db2 12.1: returns a version 4 UUID as CHAR(36), in
        # uppercase; Ibis (like Python's uuid) uses lowercase.
        # (GENERATE_UNIQUE() is not a UUID.)
        return self.f.lower(sge.Anonymous(this="GENERATE_UUID", expressions=[]))


compiler = Db2Compiler()
