from __future__ import annotations

import sqlglot.expressions as sge
from sqlglot import generator
from sqlglot.dialects import Postgres


class RisingWave(Postgres):
    # Need to disable timestamp precision
    # No "or replace" allowed in create statements
    # no "not null" clause for column constraints

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
        }
