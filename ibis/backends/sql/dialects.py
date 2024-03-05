from __future__ import annotations

import contextlib
import math

import sqlglot.expressions as sge
from sqlglot import transforms
from sqlglot.dialects import (
    TSQL,
    ClickHouse,
    Hive,
    MySQL,
    Oracle,
    Postgres,
    Snowflake,
    Spark,
    SQLite,
    Trino,
)
from sqlglot.dialects.dialect import rename_func

import ibis.backends.sql.expressions as sqlexpr

ClickHouse.Generator.TRANSFORMS |= {
    sge.ArraySize: rename_func("length"),
    sge.ArraySort: rename_func("arraySort"),
    sge.LogicalAnd: rename_func("min"),
    sge.LogicalOr: rename_func("max"),
}


class DataFusion(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            sge.Select: transforms.preprocess([transforms.eliminate_qualify]),
            sge.Pow: rename_func("pow"),
            sge.IsNan: rename_func("isnan"),
            sge.CurrentTimestamp: rename_func("now"),
            sge.CurrentDate: rename_func("today"),
            sge.Split: rename_func("string_to_array"),
            sge.Array: rename_func("make_array"),
            sge.ArrayContains: rename_func("array_has"),
            sge.ArraySize: rename_func("array_length"),
        }


class Druid(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            sge.ApproxDistinct: rename_func("approx_count_distinct"),
            sge.Pow: rename_func("power"),
        }


def _interval(self, e, quote_arg=True):
    """Work around the inability to handle string literals in INTERVAL syntax."""
    arg = e.args["this"].this
    with contextlib.suppress(AttributeError):
        arg = arg.sql(self.dialect)

    if quote_arg:
        arg = f"'{arg}'"

    return f"INTERVAL {arg} {e.args['unit']}"


class Exasol(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {sge.Interval: _interval}
        TYPE_MAPPING = Postgres.Generator.TYPE_MAPPING.copy() | {
            sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMP WITH LOCAL TIME ZONE",
        }


def _calculate_precision(interval_value: int) -> int:
    """Calculate interval precision.

    FlinkSQL interval data types use leading precision and fractional-
    seconds precision. Because the leading precision defaults to 2, we need to
    specify a different precision when the value exceeds 2 digits.

    (see
    https://learn.microsoft.com/en-us/sql/odbc/reference/appendixes/interval-literals)
    """
    # log10(interval_value) + 1 is equivalent to len(str(interval_value)), but is significantly
    # faster and more memory-efficient
    if interval_value == 0:
        return 0
    if interval_value < 0:
        raise ValueError(
            f"Expecting value to be a non-negative integer, got {interval_value}"
        )
    return int(math.log10(interval_value)) + 1


def _interval_with_precision(self, e):
    """Format interval with precision."""
    arg = e.args["this"].this
    formatted_arg = arg
    with contextlib.suppress(AttributeError):
        formatted_arg = arg.sql(self.dialect)

    unit = e.args["unit"]
    # when formatting interval scalars, need to quote arg and add precision
    if isinstance(arg, str):
        formatted_arg = f"'{formatted_arg}'"
        prec = _calculate_precision(int(arg))
        prec = max(prec, 2)
        unit += f"({prec})"

    return f"INTERVAL {formatted_arg} {unit}"


class Flink(Hive):
    class Generator(Hive.Generator):
        TYPE_MAPPING = Hive.Generator.TYPE_MAPPING.copy() | {
            sge.DataType.Type.TIME: "TIME",
        }

        TRANSFORMS = Hive.Generator.TRANSFORMS.copy() | {
            sge.Stddev: rename_func("stddev_samp"),
            sge.StddevPop: rename_func("stddev_pop"),
            sge.StddevSamp: rename_func("stddev_samp"),
            sge.Variance: rename_func("var_samp"),
            sge.VariancePop: rename_func("var_pop"),
            sge.Array: (
                lambda self,
                e: f"ARRAY[{', '.join(arg.sql(self.dialect) for arg in e.expressions)}]"
            ),
            sge.ArrayConcat: rename_func("array_concat"),
            sge.Length: rename_func("char_length"),
            sge.TryCast: lambda self,
            e: f"TRY_CAST({e.this.sql(self.dialect)} AS {e.to.sql(self.dialect)})",
            sge.DayOfYear: rename_func("dayofyear"),
            sge.DayOfWeek: rename_func("dayofweek"),
            sge.DayOfMonth: rename_func("dayofmonth"),
            sge.Interval: _interval_with_precision,
            sqlexpr.MatchRecognize: lambda self, expr: self.match_recognize_sql(expr),
            sqlexpr.MatchRecognizeDefine: lambda self, expr: self.match_recognize_define_sql(expr),
            sqlexpr.MatchRecognizeMeasure: lambda self, expr: self.match_recognize_measure(expr),
            sqlexpr.MatchRecognizeMeasures: lambda self, expr: self.match_recognize_measures_sql(expr),
            sqlexpr.MatchRecognizeOrderBy: lambda self, expr: self.match_recognize_order_by_sql(expr),
            # TODO (mehmet): Adding `match_recognize_output_mode/after_match_sql` for backends
            # separately is redundant. Using a `sqlglot.Expression` instead would eliminate this.
            sqlexpr.MatchRecognizeAfterMatch: lambda self, expr: self.match_recognize_after_match_sql(expr),
            sqlexpr.MatchRecognizeOutputMode: lambda self, expr: self.match_recognize_output_mode_sql(expr),
            sqlexpr.MatchRecognizePartitionBy: lambda self, expr: self.match_recognize_partition_by_sql(expr),
            sqlexpr.Quantifier: lambda self, expr: self.quantifier_sql(expr),
            sqlexpr.MatchRecognizePattern: lambda self, expr: self.match_recognize_pattern_sql(expr),
            sqlexpr.MatchRecognizeVariable: lambda self, expr: self.match_recognize_variable_sql(expr),
            sqlexpr.MatchRecognizeTable: lambda self, expr: self.match_recognize_table_sql(expr),
        }

        def columns_wo_table_alias(self, expr: sge.Expression) -> str | None:
            columns = expr.args.get("columns")
            if not columns:
                return None

            return ", ".join(
                f"{self.sql(column, 'this')}" for column in columns
            )

        def match_recognize_order_by_sql(self, expr: sqlexpr.MatchRecognizeOrderBy) -> str:
            if columns := self.columns_wo_table_alias(expr):
                return f"ORDER BY {columns}"
            return ""

        def match_recognize_partition_by_sql(self, expr: sqlexpr.MatchRecognizePartitionBy) -> str:
            if columns := self.columns_wo_table_alias(expr):
                return f"PARTITION BY {columns}"
            return ""

        def quantifier_sql(self, expr: sqlexpr.Quantifier) -> str:
            min_num_rows = expr.args.get("min_num_rows")
            max_num_rows = expr.args.get("max_num_rows")
            reluctant = expr.args.get("reluctant")
            if max_num_rows is None:
                if min_num_rows == 0:
                    sql = "*"
                elif min_num_rows == 1:
                    sql = "+"
                else:
                    sql = f"{{{min_num_rows},}}"

            elif min_num_rows == 0:
                if max_num_rows == 1:
                    sql = "?"
                else:
                    sql = f"{{,{max_num_rows}}}"

            elif min_num_rows == max_num_rows:
                sql = f"{{{min_num_rows}}}"

            else:
                sql = f"{{{min_num_rows},{max_num_rows}}}"

            reluctant_sql = "?" if reluctant else ""
            return f"{sql}{reluctant_sql}"

        def match_recognize_variable_sql(self, expr: sqlexpr.MatchRecognizeVariable) -> str:
            name = self.sql(expr, "name")
            definition = expr.args.get("definition")
            if not isinstance(definition, sge.Null):
                definition_sql = self.sql(definition)
                return f"{name} AS {definition_sql}"

            else:
                return name

        def match_recognize_define_sql(self, expr: sqlexpr.MatchRecognizeDefine) -> str:
            define_sql = ",".join(
                self.seg(self.sql(variable))
                for variable in expr.args.get("variables", [])
            )

            return f"DEFINE{self.indent(define_sql)}"

        def match_recognize_pattern_sql(self, expr: sqlexpr.MatchRecognizePattern) -> str:
            variable_sql_list = []
            for variable in expr.args.get("variables", []):
                name = self.sql(variable, "name")
                quantifier_sql = ""
                quantifier = variable.args.get("quantifier")
                if not isinstance(quantifier, sge.Null):
                    quantifier_sql = self.sql(variable, "quantifier")

                variable_sql_list.append(f"{name}{quantifier_sql}")

            return f"PATTERN ({' '.join(variable_sql_list)})"

        def match_recognize_measure(self, expr: sqlexpr.MatchRecognizeMeasure) -> str:
            definition_sql = self.sql(expr, "definition")
            measure_name = self.sql(expr, "name")

            return f"{definition_sql} AS {measure_name}"

        def match_recognize_measures_sql(self, expr: sqlexpr.MatchRecognizeMeasures) -> str:
            measures_sql = ",".join(
                self.seg(self.sql(measure))
                for measure in expr.args.get("measures", [])
            )

            return f"MEASURES{self.indent(measures_sql)}"

        def match_recognize_after_match_sql(self, expr: sqlexpr.MatchRecognizeAfterMatch) -> str:
            this_sql = self.sql(expr, "this")

            return f"AFTER MATCH {this_sql}"

        def match_recognize_output_mode_sql(self, expr: sqlexpr.MatchRecognizeOutputMode) -> str:
            return self.sql(expr, "this")

        def match_recognize_sql(self, expr: sqlexpr.MatchRecognize) -> str:
            partition_by_sql = self.sql(expr, "partition_by")
            order_by_sql = self.sql(expr, "order_by")
            if partition_by_sql:
                order_by_sql = self.seg(order_by_sql)

            define_sql = self.seg(self.sql(expr, "define"))
            pattern_sql = self.seg(self.sql(expr, "pattern"))
            measures_sql = self.seg(self.sql(expr, "measures"))
            output_mode_sql = self.seg(self.sql(expr, "output_mode"))
            after_match_sql = self.seg(self.sql(expr, "after_match"))

            body = "".join(
                (
                    partition_by_sql,
                    order_by_sql,
                    measures_sql,
                    output_mode_sql,
                    after_match_sql,
                    pattern_sql,
                    define_sql,
                )
            )
            alias = self.sql(expr, "alias")
            alias = f" {alias}" if alias else ""
            return f"{self.seg('MATCH_RECOGNIZE')} {self.wrap(body)}{alias}"

        def match_recognize_table_sql(self, expr: sqlexpr.MatchRecognizeTable) -> str:
            table_sql = self.sql(expr, "table")
            match_recognize_sql = self.sql(expr, "match_recognize")
            return f"{table_sql}{match_recognize_sql}"

    class Tokenizer(Hive.Tokenizer):
        # In Flink, embedded single quotes are escaped like most other SQL
        # dialects: doubling up the single quote
        #
        # We override it here because we inherit from Hive's dialect and Hive
        # uses a backslash to escape single quotes
        STRING_ESCAPES = ["'"]


class Impala(Hive):
    class Generator(Hive.Generator):
        TRANSFORMS = Hive.Generator.TRANSFORMS.copy() | {
            sge.ApproxDistinct: rename_func("ndv"),
            sge.IsNan: rename_func("is_nan"),
            sge.IsInf: rename_func("is_inf"),
            sge.DayOfWeek: rename_func("dayofweek"),
            sge.Interval: lambda self, e: _interval(self, e, quote_arg=False),
            sge.CurrentDate: rename_func("current_date"),
        }


class MSSQL(TSQL):
    class Generator(TSQL.Generator):
        TRANSFORMS = TSQL.Generator.TRANSFORMS.copy() | {
            sge.ApproxDistinct: rename_func("approx_count_distinct"),
            sge.Stddev: rename_func("stdevp"),
            sge.StddevPop: rename_func("stdevp"),
            sge.StddevSamp: rename_func("stdev"),
            sge.Variance: rename_func("var"),
            sge.VariancePop: rename_func("varp"),
            sge.Ceil: rename_func("ceiling"),
            sge.Trim: lambda self, e: f"TRIM({e.this.sql(self.dialect)})",
            sge.DateFromParts: rename_func("datefromparts"),
        }


MySQL.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.StddevPop: rename_func("stddev_pop"),
    sge.StddevSamp: rename_func("stddev_samp"),
    sge.RegexpLike: (
        lambda _, e: f"({e.this.sql('mysql')} RLIKE {e.expression.sql('mysql')})"
    ),
}


def _create_sql(self, expression: sge.Create) -> str:
    properties = expression.args.get("properties")
    temporary = any(
        isinstance(prop, sge.TemporaryProperty)
        for prop in (properties.expressions if properties else [])
    )
    kind = expression.args["kind"]
    if kind.upper() in ("TABLE", "VIEW") and temporary:
        # Force insertion of required "GLOBAL" keyword
        expression_sql = self.create_sql(expression).replace(
            "CREATE TEMPORARY", "CREATE GLOBAL TEMPORARY"
        )
        if expression.expression:  #  CREATE ... AS ...
            return self.sql(expression_sql, "expression")
        else:  #  CREATE ... ON COMMIT PRESERVE ROWS
            # Autocommit does not work here for some reason so we append it manually
            return self.sql(
                expression_sql + " ON COMMIT PRESERVE ROWS",
                "expression",
            )
    return self.create_sql(expression)


Oracle.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Create: _create_sql,
    sge.Select: transforms.preprocess([transforms.eliminate_semi_and_anti_joins]),
}

# TODO: can delete this after bumping sqlglot version > 20.9.0
Oracle.Generator.TYPE_MAPPING |= {
    sge.DataType.Type.TIMETZ: "TIME",
    sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMP",
}
Oracle.Generator.TZ_TO_WITH_TIME_ZONE = True


class Polars(Postgres):
    """Subclass of Postgres dialect for Polars.

    This is here to allow referring to the Postgres dialect as "polars"
    """


Postgres.Generator.TRANSFORMS |= {
    sge.Map: rename_func("hstore"),
    sge.Split: rename_func("string_to_array"),
    sge.RegexpSplit: rename_func("regexp_split_to_array"),
    sge.DateFromParts: rename_func("make_date"),
    sge.ArraySize: rename_func("cardinality"),
    sge.Pow: rename_func("pow"),
}


class PySpark(Spark):
    """Subclass of Spark dialect for PySpark.

    This is here to allow referring to the Spark dialect as "pyspark"
    """


class RisingWave(Postgres):
    # Need to disable timestamp precision
    # No "or replace" allowed in create statements
    # no "not null" clause for column constraints

    class Generator(Postgres.Generator):
        SINGLE_STRING_INTERVAL = True
        RENAME_TABLE_WITH_DB = False
        LOCKING_READS_SUPPORTED = True
        JOIN_HINTS = False
        TABLE_HINTS = False
        QUERY_HINTS = False
        NVL2_SUPPORTED = False
        PARAMETER_TOKEN = "$"
        TABLESAMPLE_SIZE_IS_ROWS = False
        TABLESAMPLE_SEED_KEYWORD = "REPEATABLE"
        SUPPORTS_SELECT_INTO = True
        JSON_TYPE_REQUIRED_FOR_EXTRACTION = True
        SUPPORTS_UNLOGGED_TABLES = True

        TYPE_MAPPING = Postgres.Generator.TYPE_MAPPING.copy() | {
            sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMPTZ"
        }


Snowflake.Generator.TRANSFORMS |= {
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Levenshtein: rename_func("editdistance"),
}

SQLite.Generator.TYPE_MAPPING |= {sge.DataType.Type.BOOLEAN: "BOOLEAN"}


# TODO(cpcloud): remove this hack once
# https://github.com/tobymao/sqlglot/issues/2735 is resolved
def make_cross_joins_explicit(node):
    if not (node.kind or node.side):
        node.args["kind"] = "CROSS"
    return node


Trino.Generator.TRANSFORMS |= {
    sge.BitwiseLeftShift: rename_func("bitwise_left_shift"),
    sge.BitwiseRightShift: rename_func("bitwise_right_shift"),
    sge.FirstValue: rename_func("first_value"),
    sge.Join: transforms.preprocess([make_cross_joins_explicit]),
    sge.LastValue: rename_func("last_value"),
}
