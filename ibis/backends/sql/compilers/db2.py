"""Db2 SQL compiler for Ibis expressions."""

from __future__ import annotations

import sqlglot.expressions as sge
from sqlglot import exp
from sqlglot.generator import Generator

try:
    from db2_sqlglot import Db2 as _Db2BaseDialect

    _DB2_SQLGLOT_AVAILABLE = True
except ImportError:
    _Db2BaseDialect = None
    _DB2_SQLGLOT_AVAILABLE = False

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import SQLGlotCompiler
from ibis.backends.sql.datatypes import Db2Type
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    lower_sample,
    split_select_distinct_with_order_by,
)


class Db2Generator(Generator):
    """Custom SQL generator for Db2."""

    TRANSFORMS = {
        **Generator.TRANSFORMS,
        exp.DateAdd: lambda self, e: self.func(
            "DATE_ADD", e.this, self.sql(e, "expression"), self.sql(e, "unit")
        ),
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
    }

    TYPE_MAPPING = {
        **Generator.TYPE_MAPPING,
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

    def tablesample_sql(self, expression: exp.TableSample) -> str:
        """Db2 doesn't support TABLESAMPLE, return empty string."""
        return ""

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
        """Generate CONCAT expression for Db2."""
        # Db2 uses || operator or CONCAT function
        return self.func("CONCAT", *expression.expressions)

    def substring_sql(self, expression: exp.Substring) -> str:
        """Generate SUBSTRING expression for Db2."""
        # Db2 uses SUBSTR function
        args = [expression.this]
        if expression.args.get("start"):
            args.append(expression.args["start"])
        if expression.args.get("length"):
            args.append(expression.args["length"])
        return self.func("SUBSTR", *args)

    def dateadd_sql(self, expression: exp.DateAdd) -> str:
        """Generate date addition for Db2."""
        unit = self.sql(expression, "unit")
        value = self.sql(expression, "expression")
        date = self.sql(expression, "this")

        # Db2 uses specific functions for date arithmetic
        unit_map = {
            "DAY": f"{date} + {value} DAYS",
            "MONTH": f"{date} + {value} MONTHS",
            "YEAR": f"{date} + {value} YEARS",
            "HOUR": f"{date} + {value} HOURS",
            "MINUTE": f"{date} + {value} MINUTES",
            "SECOND": f"{date} + {value} SECONDS",
        }
        return unit_map.get(unit.upper(), f"{date} + {value} {unit}")

    def extract_sql(self, expression: exp.Extract) -> str:
        """Generate EXTRACT expression for Db2."""
        # Db2 supports EXTRACT function
        return f"EXTRACT({self.sql(expression, 'this')} FROM {self.sql(expression, 'expression')})"

    def regexp_like_sql(self, expression: exp.RegexpLike) -> str:
        """Generate REGEXP_LIKE for Db2."""
        return self.func("REGEXP_LIKE", expression.this, expression.expression)

    def div_sql(self, expression: exp.Div) -> str:
        """Generate division for Db2."""
        # Db2 integer division needs special handling
        return (
            f"({self.sql(expression, 'this')} / {self.sql(expression, 'expression')})"
        )

    def mod_sql(self, expression: exp.Mod) -> str:
        """Generate modulo for Db2."""
        return self.func("MOD", expression.this, expression.expression)

    def log_sql(self, expression: exp.Log) -> str:
        """Generate logarithm for Db2."""
        if expression.args.get("base"):
            # Db2 LOG function with base
            return self.func("LOG", expression.args["base"], expression.this)
        # Natural log
        return self.func("LN", expression.this)

    def sqrt_sql(self, expression: exp.Sqrt) -> str:
        """Generate square root for Db2."""
        return self.func("SQRT", expression.this)

    def power_sql(self, expression: exp.Pow) -> str:
        """Generate power function for Db2."""
        return self.func("POWER", expression.this, expression.expression)


if _DB2_SQLGLOT_AVAILABLE:

    class Db2Dialect(_Db2BaseDialect):
        """Db2 SQL dialect for SQLGlot, extending the base dialect from db2_sqlglot."""

        class Generator(_Db2BaseDialect.Generator, Db2Generator):
            """Extended Db2 generator with Ibis-specific customisations.

            Inherits from both ``Db2BaseDialect.Generator`` (for dialect-level
            flags such as ``LIMIT_FETCH`` and correct ``limit_sql`` /
            ``offset_sql`` / ``fetch_sql``) and ``Db2Generator`` (for
            Ibis-specific TRANSFORMS and TYPE_MAPPING overrides).
            """

            TYPE_MAPPING = {
                **_Db2BaseDialect.Generator.TYPE_MAPPING,
                **Db2Generator.TYPE_MAPPING,
            }

            TRANSFORMS = {
                **_Db2BaseDialect.Generator.TRANSFORMS,
                **Db2Generator.TRANSFORMS,
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

else:
    from sqlglot.dialects.dialect import Dialect as Db2Dialect


class Db2Compiler(SQLGlotCompiler):
    """SQL compiler for the Db2 backend."""

    __slots__ = ()

    dialect = Db2Dialect
    type_mapper = Db2Type

    rewrites = (
        exclude_unsupported_window_frame_from_ops | lower_sample(),
        *SQLGlotCompiler.rewrites,
    )

    post_rewrites = (split_select_distinct_with_order_by,)

    # Exclude operations from SIMPLE_OPS that need custom implementations
    SIMPLE_OPS = {
        k: v
        for k, v in SQLGlotCompiler.SIMPLE_OPS.items()
        if k
        not in (
            ops.StartsWith,  # → visit_StartsWith (LEFT + LENGTH)
            ops.StringContains,  # → visit_StringContains (LOCATE)
            ops.RandomUUID,  # → visit_RandomUUID (GENERATE_UNIQUE)
            ops.ArgMin,  # → visit_ArgMin (raise)
            ops.ArgMax,  # → visit_ArgMax (raise)
            ops.ApproxCountDistinct,  # → visit_ApproxCountDistinct (COUNT DISTINCT)
            ops.StringSplit,  # → visit_StringSplit (raise)
            ops.Median,  # → visit_Median (already implemented)
            ops.RegexSearch,  # → visit_RegexSearch (REGEXP_LIKE)
            ops.RegexExtract,  # → visit_RegexExtract (REGEXP_SUBSTR)
            ops.RandomScalar,  # → visit_RandomScalar (RAND())
            ops.Xor,  # → visit_Xor (raise — DB2 has no XOR keyword)
        )
    }

    @staticmethod
    def _generate_groups(groups):
        """Generate GROUP BY clause."""
        return groups

    def visit_Cast(self, op, *, arg, to, **kwargs):
        """Visit a Cast operation."""
        # Db2 uses CAST syntax
        return self.cast(arg, to)

    def visit_TryCast(self, op, *, arg, to, **kwargs):
        """Visit a TryCast operation."""
        # Db2 doesn't have TRY_CAST, use CAST with error handling
        return self.cast(arg, to)

    def visit_Sample(self, op, *, table, fraction, method, seed, **kwargs):
        """Visit a Sample operation."""
        # Db2 doesn't have native TABLESAMPLE, use WHERE RAND() < fraction
        if method == "row":
            # Use RAND() function for sampling
            condition = sge.LT(
                this=sge.Anonymous(this="RAND", expressions=[]),
                expression=sge.convert(fraction),
            )
            return sge.Where(this=table, expression=condition)
        return table

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
        # Db2 uses LOCATE function
        if start is not None:
            return sge.Anonymous(this="LOCATE", expressions=[substr, arg, start])
        return sge.Anonymous(this="LOCATE", expressions=[substr, arg])

    def visit_RegexSearch(self, op, *, arg, pattern, **kwargs):
        """Visit a RegexSearch operation."""
        # Db2 uses REGEXP_LIKE function
        return sge.Anonymous(this="REGEXP_LIKE", expressions=[arg, pattern])

    def visit_RegexExtract(self, op, *, arg, pattern, index, **kwargs):
        """Visit a RegexExtract operation."""
        # Db2 uses REGEXP_SUBSTR function
        return sge.Anonymous(this="REGEXP_SUBSTR", expressions=[arg, pattern])

    def visit_RegexReplace(self, op, *, arg, pattern, replacement, **kwargs):
        """Visit a RegexReplace operation."""
        # Db2 uses REGEXP_REPLACE function
        return sge.Anonymous(
            this="REGEXP_REPLACE", expressions=[arg, pattern, replacement]
        )

    def visit_StringSplit(self, op, *, arg, delimiter):
        """Visit a StringSplit operation."""
        raise com.OperationNotDefinedError("Db2 does not support string split")

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        """Visit an ArrayCollect operation."""
        raise com.OperationNotDefinedError("Db2 does not support array collect")

    def visit_Array(self, op, *, exprs):
        """Visit an Array operation."""
        raise com.OperationNotDefinedError("Db2 does not support ARRAY literals")

    def visit_StructColumn(self, op, *, names, values):
        """Visit a StructColumn operation."""
        raise com.OperationNotDefinedError("Db2 does not support STRUCT constructor")

    def visit_Unnest(self, op, *, values):
        """Visit an Unnest operation."""
        raise com.OperationNotDefinedError("Db2 does not support UNNEST")

    def visit_Arbitrary(self, op, *, arg, where):
        """Visit an Arbitrary operation using MIN (Db2 does not have ANY_VALUE)."""
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

    def visit_Mode(self, op, *, arg, where):
        """Visit a Mode operation."""
        raise com.OperationNotDefinedError("Db2 does not support mode aggregation")

    def visit_ArgMin(self, op, *, arg, key, where):
        """Visit an ArgMin operation."""
        raise com.OperationNotDefinedError("Db2 does not support arg_min")

    def visit_ArgMax(self, op, *, arg, key, where):
        """Visit an ArgMax operation."""
        raise com.OperationNotDefinedError("Db2 does not support arg_max")

    def visit_CountDistinctStar(self, op, *, arg, where):
        """Visit a CountDistinctStar operation."""
        raise com.OperationNotDefinedError("Db2 does not support COUNT(DISTINCT *)")

    def visit_CountDistinct(self, op, *, arg, where):
        """Visit a CountDistinct operation."""
        if where is not None:
            arg = self.if_(where, arg, None)
        return sge.Count(this=sge.Distinct(expressions=[arg]))

    def visit_ApproxCountDistinct(self, op, *, arg, where):
        """Visit an ApproxCountDistinct operation."""
        # Db2 doesn't have APPROX_COUNT_DISTINCT, use COUNT(DISTINCT)
        return self.visit_CountDistinct(op, arg=arg, where=where)

    def visit_First(self, op, *, arg, where, include_null, order_by):
        """Visit a First operation."""
        raise com.OperationNotDefinedError("Db2 does not support first aggregation")

    def visit_Last(self, op, *, arg, where, include_null, order_by):
        """Visit a Last operation."""
        raise com.OperationNotDefinedError("Db2 does not support last aggregation")

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

    def visit_DateTruncate(self, op, *, arg, unit):
        """Visit a DateTruncate operation."""
        # Db2 uses TRUNC function for dates
        unit_map = {
            "Y": "YEAR",
            "Q": "QUARTER",
            "M": "MONTH",
            "W": "WEEK",
            "D": "DAY",
        }
        mapped_unit = unit_map.get(unit, unit)
        return sge.Anonymous(this="TRUNC", expressions=[arg, sge.convert(mapped_unit)])

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
        """Visit a RandomUUID operation."""
        # Db2 doesn't have native UUID generation
        # Use combination of functions to generate UUID-like string
        return sge.Anonymous(this="GENERATE_UNIQUE", expressions=[])

    def visit_Xor(self, op, *, left, right):
        """Visit a Xor operation."""
        raise com.OperationNotDefinedError("Db2 does not support the XOR operator")

    def visit_Hash(self, op, *, arg):
        """Visit a Hash operation."""
        raise com.OperationNotDefinedError("Db2 does not support hash")

    def visit_HashBytes(self, op, *, arg, how):
        """Visit a HashBytes operation."""
        raise com.OperationNotDefinedError("Db2 does not support hash_bytes")

    def visit_InMemoryTable(self, op, *, name, schema, data):
        # DB2 stores all identifiers uppercase — memtable names must match
        return super().visit_InMemoryTable(
            op, name=name.upper(), schema=schema, data=data
        )


compiler = Db2Compiler()
