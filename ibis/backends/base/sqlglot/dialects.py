from __future__ import annotations

import contextlib

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
from sqlglot.dialects.dialect import create_with_partitions_sql, rename_func

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
        }

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
    # TODO: should we use CREATE PRIVATE instead?  That will set an implicit
    # lower bound of Oracle 18c
    properties = expression.args.get("properties")
    temporary = any(
        isinstance(prop, sge.TemporaryProperty)
        for prop in (properties.expressions if properties else [])
    )

    kind = expression.args["kind"]
    if (obj := kind.upper()) in ("TABLE", "VIEW") and temporary:
        if expression.expression:
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} AS {self.sql(expression, 'expression')}"
        else:
            # TODO: why does autocommit not work here?  need to specify the ON COMMIT part...
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} ON COMMIT PRESERVE ROWS"

    return create_with_partitions_sql(self, expression)


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
    sge.Join: transforms.preprocess([make_cross_joins_explicit]),
}
