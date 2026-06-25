"""DB2 SQL compiler for Ibis expressions."""

from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.sql.compilers.base import SQLGlotCompiler
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    lower_sample,
)
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.generator import Generator

import ibis.expr.operations as ops
from db2_sqlglot import Db2 as DB2BaseDialect

# Import type mapper from local datatypes module
from ibis.backends.db2.datatypes import ibis_type_to_db2_type


class DB2Generator(Generator):
    """Custom SQL generator for DB2."""

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
            self.sql(e, "separator") if e.args.get("separator") else self.sql(exp.Literal.string(",")),
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

    def limit_sql(self, expression: exp.Limit, top: bool = False) -> str:
        """Generate LIMIT clause for DB2."""
        if top:
            return f"TOP {self.sql(expression, 'expression')}"
        return f"FETCH FIRST {self.sql(expression, 'expression')} ROWS ONLY"

    def offset_sql(self, expression: exp.Offset) -> str:
        """Generate OFFSET clause for DB2."""
        return f"OFFSET {self.sql(expression, 'expression')} ROWS"
    
    def fetch_sql(self, expression: exp.Fetch) -> str:
        """Generate FETCH clause for DB2."""
        return f"FETCH FIRST {self.sql(expression, 'count')} ROWS ONLY"

    def select_sql(self, expression: exp.Select) -> str:
        """Generate SELECT statement for DB2 with proper clause ordering."""
        import re
        
        # Check if we have both OFFSET and LIMIT
        has_offset = expression.args.get("offset") is not None
        has_limit = expression.args.get("limit") is not None
        
        if has_offset and has_limit:
            # Temporarily remove both limit and offset to generate SQL without them
            limit_expr = expression.args.pop("limit")
            offset_expr = expression.args.pop("offset")
            
            # Generate SQL with everything except limit and offset
            sql = super().select_sql(expression)
            
            # Restore limit and offset
            expression.args["limit"] = limit_expr
            expression.args["offset"] = offset_expr
            
            # Now manually add OFFSET and FETCH FIRST in the correct order
            offset_val = self.sql(offset_expr, "expression")
            limit_val = self.sql(limit_expr, "expression")
            
            sql = f"{sql} OFFSET {offset_val} ROWS FETCH FIRST {limit_val} ROWS ONLY"
        else:
            # Normal generation for cases without both OFFSET and LIMIT
            sql = super().select_sql(expression)
        
        # Fix spacing issues in generated SQL
        # DB2 requires proper spacing between clauses
        sql = sql.replace("LASTFETCH", "LAST FETCH")
        sql = sql.replace("DESCFETCH", "DESC FETCH")
        sql = sql.replace("ASCFETCH", "ASC FETCH")
        sql = sql.replace("ONLYOFFSET", "ONLY OFFSET")
        sql = re.sub(r'(\d+)\s*ROWSFETCH', r'\1 ROWS FETCH', sql)
        
        # Ensure space before FETCH FIRST and OFFSET
        sql = re.sub(r'(\S)FETCH\s+FIRST', r'\1 FETCH FIRST', sql)
        sql = re.sub(r'(\S)OFFSET\s+', r'\1 OFFSET ', sql)
        
        # Remove ROWS BETWEEN clauses from window functions that don't support them
        sql = re.sub(
            r'(ROW_NUMBER\(\)|RANK\(\)|DENSE_RANK\(\)|PERCENT_RANK\(\)|CUME_DIST\(\)|NTILE\([^)]+\)|LAG\([^)]+\)|LEAD\([^)]+\)|FIRST_VALUE\([^)]+\)|LAST_VALUE\([^)]+\))\s+OVER\s+\(([^)]*?)\s+ROWS\s+BETWEEN[^)]+\)',
            r'\1 OVER (\2)',
            sql
        )
        
        return sql

    def tablesample_sql(self, expression: exp.TableSample) -> str:
        """DB2 doesn't support TABLESAMPLE, return empty string."""
        return ""

    def cast_sql(self, expression: exp.Cast) -> str:
        """Generate CAST expression for DB2."""
        return f"CAST({self.sql(expression, 'this')} AS {self.sql(expression, 'to')})"

    def trycast_sql(self, expression: exp.TryCast) -> str:
        """DB2 doesn't have TRY_CAST, use regular CAST."""
        return self.cast_sql(expression)

    def boolean_sql(self, expression: exp.Boolean) -> str:
        """Generate boolean literal for DB2."""
        return "TRUE" if expression.this else "FALSE"

    def concat_sql(self, expression: exp.Concat) -> str:
        """Generate CONCAT expression for DB2."""
        # DB2 uses || operator or CONCAT function
        return self.func("CONCAT", *expression.expressions)

    def substring_sql(self, expression: exp.Substring) -> str:
        """Generate SUBSTRING expression for DB2."""
        # DB2 uses SUBSTR function
        args = [expression.this]
        if expression.args.get("start"):
            args.append(expression.args["start"])
        if expression.args.get("length"):
            args.append(expression.args["length"])
        return self.func("SUBSTR", *args)

    def dateadd_sql(self, expression: exp.DateAdd) -> str:
        """Generate date addition for DB2."""
        unit = self.sql(expression, "unit")
        value = self.sql(expression, "expression")
        date = self.sql(expression, "this")
        
        # DB2 uses specific functions for date arithmetic
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
        """Generate EXTRACT expression for DB2."""
        # DB2 supports EXTRACT function
        return f"EXTRACT({self.sql(expression, 'this')} FROM {self.sql(expression, 'expression')})"

    def regexp_like_sql(self, expression: exp.RegexpLike) -> str:
        """Generate REGEXP_LIKE for DB2."""
        return self.func("REGEXP_LIKE", expression.this, expression.expression)

    def if_sql(self, expression: exp.If) -> str:
        """Generate IF/CASE expression for DB2."""
        # DB2 uses CASE WHEN for conditional logic
        return f"CASE WHEN {self.sql(expression, 'this')} THEN {self.sql(expression, 'true')} ELSE {self.sql(expression, 'false')} END"

    def div_sql(self, expression: exp.Div) -> str:
        """Generate division for DB2."""
        # DB2 integer division needs special handling
        return f"({self.sql(expression, 'this')} / {self.sql(expression, 'expression')})"

    def mod_sql(self, expression: exp.Mod) -> str:
        """Generate modulo for DB2."""
        return self.func("MOD", expression.this, expression.expression)

    def log_sql(self, expression: exp.Log) -> str:
        """Generate logarithm for DB2."""
        if expression.args.get("base"):
            # DB2 LOG function with base
            return self.func("LOG", expression.args["base"], expression.this)
        # Natural log
        return self.func("LN", expression.this)

    def sqrt_sql(self, expression: exp.Sqrt) -> str:
        """Generate square root for DB2."""
        return self.func("SQRT", expression.this)

    def power_sql(self, expression: exp.Pow) -> str:
        """Generate power function for DB2."""
        return self.func("POWER", expression.this, expression.expression)


class DB2Dialect(DB2BaseDialect):
    """DB2 SQL dialect for SQLGlot, extending the base dialect from sqlglot-db2."""

    class Generator(DB2Generator):
        """Extended DB2 generator with Ibis-specific customizations."""
        
        # Inherit from both the base DB2 generator and our custom generator
        TYPE_MAPPING = {
            **DB2BaseDialect.Generator.TYPE_MAPPING,
            **DB2Generator.TYPE_MAPPING,
        }
        
        TRANSFORMS = {
            **DB2BaseDialect.Generator.TRANSFORMS,
            **DB2Generator.TRANSFORMS,
        }

    class Parser(DB2BaseDialect.Parser):
        """Extended DB2 parser with Ibis-specific customizations."""
        
        FUNCTIONS = {
            **DB2BaseDialect.Parser.FUNCTIONS,
            "LOCATE": exp.StrPosition.from_arg_list,
            "LISTAGG": exp.GroupConcat.from_arg_list,
            "DAYS_BETWEEN": exp.DateDiff.from_arg_list,
            "REGEXP_LIKE": lambda args: exp.RegexpLike(
                this=args[0], expression=args[1]
            ),
        }


class DB2Compiler(SQLGlotCompiler):
    """SQL compiler for DB2 backend."""

    __slots__ = ()

    dialect = DB2Dialect
    type_mapper = ibis_type_to_db2_type

    rewrites = (
        exclude_unsupported_window_frame_from_ops | lower_sample(),
        *SQLGlotCompiler.rewrites,
    )
    
    # Exclude StartsWith and StringContains from SIMPLE_OPS to use our custom implementations
    # StartsWith: base class maps to "starts_with" function which doesn't exist in DB2
    # StringContains: base class maps to "contains" function which requires text search feature
    SIMPLE_OPS = {
        k: v for k, v in SQLGlotCompiler.SIMPLE_OPS.items()
        if k not in (ops.StartsWith, ops.StringContains)
    }

    @staticmethod
    def _generate_groups(groups):
        """Generate GROUP BY clause."""
        return groups

    def visit_Cast(self, op, *, arg, to, **kwargs):
        """Visit a Cast operation."""
        # DB2 uses CAST syntax
        return self.cast(arg, to)

    def visit_TryCast(self, op, *, arg, to, **kwargs):
        """Visit a TryCast operation."""
        # DB2 doesn't have TRY_CAST, use CAST with error handling
        return self.cast(arg, to)

    def visit_Sample(self, op, *, table, fraction, method, seed, **kwargs):
        """Visit a Sample operation."""
        # DB2 doesn't have native TABLESAMPLE, use WHERE RAND() < fraction
        if method == "row":
            # Use RAND() function for sampling
            condition = sge.LT(
                this=sge.Anonymous(this="RAND", expressions=[]),
                expression=sge.convert(fraction)
            )
            return sge.Where(this=table, expression=condition)
        return table

    def visit_StringContains(self, op, *, haystack, needle, **kwargs):
        """Visit a StringContains operation."""
        # DB2 uses LOCATE function
        return sge.GT(
            this=sge.Anonymous(
                this="LOCATE",
                expressions=[needle, haystack]
            ),
            expression=sge.convert(0)
        )

    def visit_EndsWith(self, op, *, arg, end, **kwargs):
        """Visit an EndsWith operation.
        
        Generates SQL: arg LIKE '%' || REPLACE(REPLACE(end, '%', '!%'), '_', '!_') ESCAPE '!'
        This escapes LIKE wildcards (% and _) to match them literally.
        """
        # Escape LIKE wildcards in the pattern using ! as escape char
        # First replace % with !%, then replace _ with !_
        escaped = sge.Anonymous(
            this="REPLACE",
            expressions=[
                sge.Anonymous(
                    this="REPLACE",
                    expressions=[end, sge.convert('%'), sge.convert('!%')]
                ),
                sge.convert('_'),
                sge.convert('!_')
            ]
        )
        # Create pattern: '%' || escaped_end
        pattern = sge.DPipe(this=sge.convert('%'), expression=escaped)
        # Return LIKE with ESCAPE clause
        return sge.Escape(
            this=sge.Like(this=arg, expression=pattern),
            expression=sge.convert('!')
        )

    def visit_StartsWith(self, op, *, arg, start, **kwargs):
        """Visit a StartsWith operation.
        
        Generates SQL: arg LIKE REPLACE(REPLACE(start, '%', '!%'), '_', '!_') || '%' ESCAPE '!'
        This escapes LIKE wildcards (% and _) to match them literally.
        """
        # Escape LIKE wildcards in the pattern using ! as escape char
        # First replace % with !%, then replace _ with !_
        escaped = sge.Anonymous(
            this="REPLACE",
            expressions=[
                sge.Anonymous(
                    this="REPLACE",
                    expressions=[start, sge.convert('%'), sge.convert('!%')]
                ),
                sge.convert('_'),
                sge.convert('!_')
            ]
        )
        # Create pattern: escaped_start || '%'
        pattern = sge.DPipe(this=escaped, expression=sge.convert('%'))
        # Return LIKE with ESCAPE clause
        return sge.Escape(
            this=sge.Like(this=arg, expression=pattern),
            expression=sge.convert('!')
        )

    def visit_StringFind(self, op, *, arg, substr, start, end, **kwargs):
        """Visit a StringFind operation."""
        # DB2 uses LOCATE function
        if start is not None:
            return sge.Anonymous(
                this="LOCATE",
                expressions=[substr, arg, start]
            )
        return sge.Anonymous(
            this="LOCATE",
            expressions=[substr, arg]
        )

    def visit_RegexSearch(self, op, *, arg, pattern, **kwargs):
        """Visit a RegexSearch operation."""
        # DB2 uses REGEXP_LIKE function
        return sge.Anonymous(
            this="REGEXP_LIKE",
            expressions=[arg, pattern]
        )

    def visit_RegexExtract(self, op, *, arg, pattern, index, **kwargs):
        """Visit a RegexExtract operation."""
        # DB2 uses REGEXP_SUBSTR function
        return sge.Anonymous(
            this="REGEXP_SUBSTR",
            expressions=[arg, pattern]
        )

    def visit_RegexReplace(self, op, *, arg, pattern, replacement, **kwargs):
        """Visit a RegexReplace operation."""
        # DB2 uses REGEXP_REPLACE function
        return sge.Anonymous(
            this="REGEXP_REPLACE",
            expressions=[arg, pattern, replacement]
        )

    def visit_StringSplit(self, op, *, arg, delimiter):
        """Visit a StringSplit operation."""
        # DB2 doesn't have native string split, return as-is
        # This would need to be handled at a higher level
        return arg

    def visit_ArrayCollect(self, op, *, arg):
        """Visit an ArrayCollect operation."""
        # DB2 uses LISTAGG for array aggregation
        return sge.Anonymous(
            this="LISTAGG",
            expressions=[arg, sge.convert(",")]
        )

    def visit_Median(self, op, *, arg, where):
        """Visit a Median operation."""
        # DB2 uses MEDIAN function
        func = sge.Anonymous(this="MEDIAN", expressions=[arg])
        if where is not None:
            return sge.Filter(this=func, expression=sge.Where(this=where))
        return func

    def visit_Mode(self, op, *, arg, where):
        """Visit a Mode operation."""
        # DB2 doesn't have native MODE, use workaround with COUNT and GROUP BY
        # This is a simplified version
        return sge.Anonymous(this="MODE", expressions=[arg])

    def visit_ArgMin(self, op, *, arg, key, where):
        """Visit an ArgMin operation."""
        # DB2 uses MIN_BY or equivalent window function
        return sge.Anonymous(this="MIN_BY", expressions=[arg, key])

    def visit_ArgMax(self, op, *, arg, key, where):
        """Visit an ArgMax operation."""
        # DB2 uses MAX_BY or equivalent window function
        return sge.Anonymous(this="MAX_BY", expressions=[arg, key])

    def visit_CountDistinct(self, op, *, arg, where):
        """Visit a CountDistinct operation."""
        func = sge.Count(this=arg, distinct=True)
        if where is not None:
            return sge.Filter(this=func, expression=sge.Where(this=where))
        return func

    def visit_ApproxCountDistinct(self, op, *, arg, where):
        """Visit an ApproxCountDistinct operation."""
        # DB2 doesn't have APPROX_COUNT_DISTINCT, use COUNT(DISTINCT)
        return self.visit_CountDistinct(op, arg=arg, where=where)

    def visit_First(self, op, *, arg, where):
        """Visit a First operation."""
        # DB2 uses FIRST_VALUE window function
        return sge.Anonymous(this="FIRST_VALUE", expressions=[arg])

    def visit_Last(self, op, *, arg, where):
        """Visit a Last operation."""
        # DB2 uses LAST_VALUE window function
        return sge.Anonymous(this="LAST_VALUE", expressions=[arg])

    def visit_Lag(self, op, *, arg, offset, default):
        """Visit a Lag operation."""
        # DB2 LAG function
        expressions = [arg]
        if offset is not None:
            expressions.append(offset)
        if default is not None:
            expressions.append(default)
        return sge.Anonymous(this="LAG", expressions=expressions)

    def visit_Lead(self, op, *, arg, offset, default):
        """Visit a Lead operation."""
        # DB2 LEAD function
        expressions = [arg]
        if offset is not None:
            expressions.append(offset)
        if default is not None:
            expressions.append(default)
        return sge.Anonymous(this="LEAD", expressions=expressions)

    def visit_RowNumber(self, op):
        """Visit a RowNumber operation."""
        # DB2 ROW_NUMBER function
        return sge.Anonymous(this="ROW_NUMBER", expressions=[])

    def visit_Rank(self, op):
        """Visit a Rank operation."""
        # DB2 RANK function
        return sge.Anonymous(this="RANK", expressions=[])

    def visit_DenseRank(self, op):
        """Visit a DenseRank operation."""
        # DB2 DENSE_RANK function
        return sge.Anonymous(this="DENSE_RANK", expressions=[])

    def visit_PercentRank(self, op):
        """Visit a PercentRank operation."""
        # DB2 PERCENT_RANK function
        return sge.Anonymous(this="PERCENT_RANK", expressions=[])

    def visit_CumeDist(self, op):
        """Visit a CumeDist operation."""
        # DB2 CUME_DIST function
        return sge.Anonymous(this="CUME_DIST", expressions=[])

    def visit_NTile(self, op, *, buckets):
        """Visit an NTile operation."""
        # DB2 NTILE function
        return sge.Anonymous(this="NTILE", expressions=[buckets])

    def visit_DateTruncate(self, op, *, arg, unit):
        """Visit a DateTruncate operation."""
        # DB2 uses TRUNC function for dates
        unit_map = {
            "Y": "YEAR",
            "Q": "QUARTER",
            "M": "MONTH",
            "W": "WEEK",
            "D": "DAY",
        }
        db2_unit = unit_map.get(unit, unit)
        return sge.Anonymous(
            this="TRUNC",
            expressions=[arg, sge.convert(db2_unit)]
        )

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
        # DB2 uses DAYOFWEEK function (1=Sunday, 7=Saturday)
        # Adjust to 0-based index (0=Monday)
        return sge.Sub(
            this=sge.Anonymous(this="DAYOFWEEK", expressions=[arg]),
            expression=sge.convert(1)
        )

    def visit_DayOfWeekName(self, op, *, arg):
        """Visit a DayOfWeekName operation."""
        # DB2 uses DAYNAME function
        return sge.Anonymous(this="DAYNAME", expressions=[arg])

    def visit_RandomScalar(self, op):
        """Visit a RandomScalar operation."""
        # DB2 uses RAND() function
        return sge.Anonymous(this="RAND", expressions=[])

    def visit_RandomUUID(self, op):
        """Visit a RandomUUID operation."""
        # DB2 doesn't have native UUID generation
        # Use combination of functions to generate UUID-like string
        return sge.Anonymous(this="GENERATE_UNIQUE", expressions=[])

    def visit_Hash(self, op, *, arg):
        """Visit a Hash operation."""
        # DB2 uses HASH function
        return sge.Anonymous(this="HASH", expressions=[arg])

    def visit_HashBytes(self, op, *, arg, how):
        """Visit a HashBytes operation."""
        # DB2 uses HASH function with algorithm
        return sge.Anonymous(this="HASH", expressions=[arg, sge.convert(how)])

