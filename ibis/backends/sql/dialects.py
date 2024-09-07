from __future__ import annotations

import contextlib
import math
from copy import deepcopy

import sqlglot.expressions as sge
import sqlglot.generator as sgn
from sqlglot import transforms
from sqlglot.dialects import (
    TSQL,
    Hive,
    MySQL,
    Oracle,
    Postgres,
    Snowflake,
    Spark,
    SQLite,
    Trino,
)
from sqlglot.dialects import ClickHouse as _ClickHouse
from sqlglot.dialects.dialect import rename_func
from sqlglot.helper import find_new_name, seq_get


class ClickHouse(_ClickHouse):
    class Generator(_ClickHouse.Generator):
        _ClickHouse.Generator.TRANSFORMS |= {
            sge.ArraySize: rename_func("length"),
            sge.ArraySort: rename_func("arraySort"),
            sge.LogicalAnd: rename_func("min"),
            sge.LogicalOr: rename_func("max"),
        }

        def except_op(self, expression: sge.Except) -> str:
            return f"EXCEPT{' DISTINCT' if expression.args.get('distinct') else ' ALL'}"

        def intersect_op(self, expression: sge.Intersect) -> str:
            return (
                f"INTERSECT{' DISTINCT' if expression.args.get('distinct') else ' ALL'}"
            )


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


def _group_concat(self, e):
    this = self.sql(e, "this")
    separator = self.sql(e, "separator") or "','"
    return f"GROUP_CONCAT({this} SEPARATOR {separator})"


class Exasol(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            sge.Interval: _interval,
            sge.GroupConcat: _group_concat,
            sge.ApproxDistinct: rename_func("approximate_count_distinct"),
        }
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
        unit.args["this"] += f"({prec})"

    return f"INTERVAL {formatted_arg} {unit}"


def _explode_to_unnest():
    """Convert explode into unnest.

    NOTE: Flink doesn't support UNNEST WITH ORDINALITY or UNNEST WITH OFFSET.
    """

    def _explode_to_unnest(expression: sge.Expression) -> sge.Expression:
        if isinstance(expression, sge.Select):
            from sqlglot.optimizer.scope import Scope

            taken_select_names = set(expression.named_selects)
            taken_source_names = {name for name, _ in Scope(expression).references}

            def new_name(names: set[str], name: str) -> str:
                name = find_new_name(names, name)
                names.add(name)
                return name

            # we use list here because expression.selects is mutated inside the loop
            for select in list(expression.selects):
                explode = select.find(sge.Explode)

                if explode:
                    explode_alias = ""

                    if isinstance(select, sge.Alias):
                        explode_alias = select.args["alias"]
                        alias = select
                    elif isinstance(select, sge.Aliases):
                        explode_alias = select.aliases[1]
                        alias = select.replace(sge.alias_(select.this, "", copy=False))
                    else:
                        alias = select.replace(sge.alias_(select, ""))
                        explode = alias.find(sge.Explode)
                        assert explode

                    explode_arg = explode.this

                    # This ensures that we won't use EXPLODE's argument as a new selection
                    if isinstance(explode_arg, sge.Column):
                        taken_select_names.add(explode_arg.output_name)

                    unnest_source_alias = new_name(taken_source_names, "_u")

                    if not explode_alias:
                        explode_alias = new_name(taken_select_names, "col")

                    alias.set("alias", sge.to_identifier(explode_alias))

                    column = sge.column(explode_alias, table=unnest_source_alias)

                    explode.replace(column)

                    expression.join(
                        sge.alias_(
                            sge.Unnest(
                                expressions=[explode_arg.copy()],
                            ),
                            unnest_source_alias,
                            table=[explode_alias],
                        ),
                        join_type="CROSS",
                        copy=False,
                    )

        return expression

    return _explode_to_unnest


class Flink(Hive):
    UNESCAPED_SEQUENCES = {"\\\\d": "\\d"}
    REGEXP_EXTRACT_DEFAULT_GROUP = 0

    class Generator(Hive.Generator):
        UNNEST_WITH_ORDINALITY = False

        TYPE_MAPPING = Hive.Generator.TYPE_MAPPING.copy() | {
            sge.DataType.Type.TIME: "TIME",
            sge.DataType.Type.STRUCT: "ROW",
        }

        TRANSFORMS = Hive.Generator.TRANSFORMS.copy() | {
            sge.Select: transforms.preprocess([_explode_to_unnest()]),
            sge.Stddev: rename_func("stddev_samp"),
            sge.StddevPop: rename_func("stddev_pop"),
            sge.StddevSamp: rename_func("stddev_samp"),
            sge.Variance: rename_func("var_samp"),
            sge.VariancePop: rename_func("var_pop"),
            sge.ArrayConcat: rename_func("array_concat"),
            sge.ArraySize: rename_func("cardinality"),
            sge.ArrayAgg: rename_func("array_agg"),
            sge.ArraySort: rename_func("array_sort"),
            sge.Length: rename_func("char_length"),
            sge.TryCast: lambda self,
            e: f"TRY_CAST({e.this.sql(self.dialect)} AS {e.to.sql(self.dialect)})",
            sge.DayOfYear: rename_func("dayofyear"),
            sge.DayOfWeek: rename_func("dayofweek"),
            sge.DayOfMonth: rename_func("dayofmonth"),
            sge.Interval: _interval_with_precision,
        }

        # Flink is like Hive except where it might actually be convenient
        #
        # UNNEST works like the SQL standard, and not like Hive, so we have to
        # override sqlglot here and convince it that flink is not like Hive
        # when it comes to unnesting
        TRANSFORMS.pop(sge.Unnest, None)

        def unnest_sql(self, expression: sge.Unnest) -> str:
            return sgn.Generator.unnest_sql(self, expression)

        def struct_sql(self, expression: sge.Struct) -> str:
            from sqlglot.optimizer.annotate_types import annotate_types

            expression = annotate_types(expression)

            values = []
            schema = []

            for e in expression.expressions:
                if isinstance(e, sge.PropertyEQ):
                    e = sge.alias_(e.expression, e.this)
                # named structs
                if isinstance(e, sge.Alias):
                    if e.type and e.type.is_type(sge.DataType.Type.UNKNOWN):
                        self.unsupported(
                            "Cannot convert untyped key-value definitions (try annotate_types)."
                        )
                    else:
                        schema.append(f"{self.sql(e, 'alias')} {self.sql(e.type)}")
                    values.append(self.sql(e, "this"))
                else:
                    values.append(self.sql(e))

            if not (size := len(expression.expressions)) or len(schema) != size:
                return self.func("ROW", *values)
            return f"CAST(ROW({', '.join(values)}) AS ROW({', '.join(schema)}))"

        def array_sql(self, expression: sge.Array) -> str:
            # workaround for the time being because you cannot construct an array of named
            # STRUCTs directly from the ARRAY[] constructor
            # https://issues.apache.org/jira/browse/FLINK-34898
            from sqlglot.optimizer.annotate_types import annotate_types

            expression = annotate_types(expression)
            first_arg = seq_get(expression.expressions, 0)
            # it's an array of structs
            if isinstance(first_arg, sge.Struct):
                # get rid of aliasing because we want to compile this as CAST instead
                args = deepcopy(expression.expressions)
                for arg in args:
                    for e in arg.expressions:
                        arg.set("expressions", [e.unalias() for e in arg.expressions])

                format_values = ", ".join([self.sql(arg) for arg in args])
                # all elements of the array should have the same type
                format_dtypes = self.sql(first_arg.type)

                return f"CAST(ARRAY[{format_values}] AS ARRAY<{format_dtypes}>)"

            return (
                f"ARRAY[{', '.join(self.sql(arg) for arg in expression.expressions)}]"
            )

    class Tokenizer(Hive.Tokenizer):
        # In Flink, embedded single quotes are escaped like most other SQL
        # dialects: doubling up the single quote
        #
        # We override it here because we inherit from Hive's dialect and Hive
        # uses a backslash to escape single quotes
        STRING_ESCAPES = ["'"]


class Impala(Hive):
    NULL_ORDERING = "nulls_are_large"
    REGEXP_EXTRACT_DEFAULT_GROUP = 0

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


# hack around https://github.com/tobymao/sqlglot/issues/3684
Oracle.NULL_ORDERING = "nulls_are_large"
Oracle.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Create: _create_sql,
    sge.Select: transforms.preprocess(
        [
            transforms.eliminate_semi_and_anti_joins,
            transforms.eliminate_distinct_on,
            transforms.eliminate_qualify,
        ]
    ),
    sge.GroupConcat: rename_func("listagg"),
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
        PARAMETER_TOKEN = "$"  # noqa: S105
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


Trino.Generator.TRANSFORMS |= {
    sge.BitwiseLeftShift: rename_func("bitwise_left_shift"),
    sge.BitwiseRightShift: rename_func("bitwise_right_shift"),
    sge.FirstValue: rename_func("first_value"),
    sge.LastValue: rename_func("last_value"),
}
