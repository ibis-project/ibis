from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AthenaCompiler",
    "BigQueryCompiler",
    "ClickHouseCompiler",
    "DataFusionCompiler",
    "DatabricksCompiler",
    "DruidCompiler",
    "DuckDBCompiler",
    "ExasolCompiler",
    "FlinkCompiler",
    "ImpalaCompiler",
    "MSSQLCompiler",
    "MaterializeCompiler",
    "MySQLCompiler",
    "OracleCompiler",
    "PostgresCompiler",
    "PySparkCompiler",
    "RisingWaveCompiler",
    "SQLiteCompiler",
    "SnowflakeCompiler",
    "TrinoCompiler",
]

_COMPILER_MODULES = {
    "AthenaCompiler": "ibis.backends.sql.compilers.athena",
    "BigQueryCompiler": "ibis.backends.sql.compilers.bigquery",
    "ClickHouseCompiler": "ibis.backends.sql.compilers.clickhouse",
    "DataFusionCompiler": "ibis.backends.sql.compilers.datafusion",
    "DatabricksCompiler": "ibis.backends.sql.compilers.databricks",
    "DruidCompiler": "ibis.backends.sql.compilers.druid",
    "DuckDBCompiler": "ibis.backends.sql.compilers.duckdb",
    "ExasolCompiler": "ibis.backends.sql.compilers.exasol",
    "FlinkCompiler": "ibis.backends.sql.compilers.flink",
    "ImpalaCompiler": "ibis.backends.sql.compilers.impala",
    "MaterializeCompiler": "ibis.backends.sql.compilers.materialize",
    "MSSQLCompiler": "ibis.backends.sql.compilers.mssql",
    "MySQLCompiler": "ibis.backends.sql.compilers.mysql",
    "OracleCompiler": "ibis.backends.sql.compilers.oracle",
    "PostgresCompiler": "ibis.backends.sql.compilers.postgres",
    "PySparkCompiler": "ibis.backends.sql.compilers.pyspark",
    "RisingWaveCompiler": "ibis.backends.sql.compilers.risingwave",
    "SQLiteCompiler": "ibis.backends.sql.compilers.sqlite",
    "SnowflakeCompiler": "ibis.backends.sql.compilers.snowflake",
    "TrinoCompiler": "ibis.backends.sql.compilers.trino",
}


def __getattr__(name: str) -> Any:
    if (module_name := _COMPILER_MODULES.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_COMPILER_MODULES))
