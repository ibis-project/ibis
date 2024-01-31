from __future__ import annotations

import sqlglot.expressions as sge
from sqlglot import generator
from sqlglot.dialects import Postgres

# from sqlglot.dialects.dialect import rename_func


def _datatype_sql(self: generator.Generator, expression: sge.DataType) -> str:
    if expression.is_type("timestamptz"):
        # Use this to strip off timestamp precision
        return "TIMESTAMPTZ"
    if expression.is_type("decimal"):
        # Risingwave doesn't allow specifying precision
        # Also max precision (or default precision) is 24
        return "NUMERIC"
    return self.datatype_sql(expression)


class RisingWave(Postgres):
    # Need to disable timestamp precision
    # No "or replace" allowed in create statements
    # no "not null" clause for column constraints
    #
    #

    class Generator(generator.Generator):
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

        TYPE_MAPPING = {
            **Postgres.Generator.TYPE_MAPPING,
            sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMPTZ",
        }

        TRANSFORMS = {
            **Postgres.Generator.TRANSFORMS,
            sge.DataType: _datatype_sql,
        }
