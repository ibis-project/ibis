from __future__ import annotations

import contextlib
import decimal
import math
import operator
from operator import and_, lshift, or_, rshift, xor

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis import _
from ibis import literal as L
from ibis.expr import datatypes as dt
from ibis.tests.util import assert_equal

try:
    import duckdb

    DuckDBConversionException = duckdb.ConversionException
except ImportError:
    duckdb = None
    DuckDBConversionException = None


try:
    import clickhouse_connect as cc

    ClickhouseDriverOperationalError = cc.driver.ProgrammingError
except ImportError:
    ClickhouseDriverOperationalError = None

try:
    from py4j.protocol import Py4JError
except ImportError:
    Py4JError = None

try:
    from pyarrow import ArrowNotImplementedError, ArrowTypeError
except ImportError:
    ArrowTypeError = None

try:
    from google.api_core.exceptions import BadRequest as GoogleBadRequest
except ImportError:
    GoogleBadRequest = None

try:
    from impala.error import HiveServer2Error as ImpalaHiveServer2Error
except ImportError:
    ImpalaHiveServer2Error = None


@pytest.mark.parametrize(
    ("expr", "expected_types"),
    [
        param(
            ibis.literal(1, type=dt.int8),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "TINYINT",
                "postgres": "integer",
                "flink": "TINYINT NOT NULL",
            },
            id="int8",
        ),
        param(
            ibis.literal(1, type=dt.int16),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "SMALLINT",
                "postgres": "integer",
                "flink": "SMALLINT NOT NULL",
            },
            id="int16",
        ),
        param(
            ibis.literal(1, type=dt.int32),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "flink": "INT NOT NULL",
            },
            id="int32",
        ),
        param(
            ibis.literal(1, type=dt.int64),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "BIGINT",
                "postgres": "integer",
                "flink": "BIGINT NOT NULL",
            },
            id="int64",
        ),
        param(
            ibis.literal(1, type=dt.uint8),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "UTINYINT",
                "postgres": "integer",
                "flink": "TINYINT NOT NULL",
            },
            id="uint8",
        ),
        param(
            ibis.literal(1, type=dt.uint16),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "USMALLINT",
                "postgres": "integer",
                "flink": "SMALLINT NOT NULL",
            },
            id="uint16",
        ),
        param(
            ibis.literal(1, type=dt.uint32),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "UINTEGER",
                "postgres": "integer",
                "flink": "INT NOT NULL",
            },
            id="uint32",
        ),
        param(
            ibis.literal(1, type=dt.uint64),
            {
                "bigquery": "INT64",
                "clickhouse": "UInt8",
                "impala": "TINYINT",
                "snowflake": "INTEGER",
                "sqlite": "integer",
                "trino": "integer",
                "duckdb": "UBIGINT",
                "postgres": "integer",
                "flink": "BIGINT NOT NULL",
            },
            id="uint64",
        ),
        param(
            ibis.literal(1, type=dt.float16),
            {
                "bigquery": "FLOAT64",
                "clickhouse": "Float64",
                "impala": "DECIMAL(2,1)",
                "snowflake": "INTEGER",
                "sqlite": "real",
                "trino": "double",
                "duckdb": "FLOAT",
                "postgres": "numeric",
                "flink": "FLOAT NOT NULL",
            },
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    "Unsupported type: Float16(nullable=True)",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ["datafusion"],
                    "Expected np.float16 instance",
                    raises=ArrowNotImplementedError,
                ),
            ],
            id="float16",
        ),
        param(
            ibis.literal(1, type=dt.float32),
            {
                "bigquery": "FLOAT64",
                "clickhouse": "Float64",
                "impala": "DECIMAL(2,1)",
                "snowflake": "INTEGER",
                "sqlite": "real",
                "trino": "double",
                "duckdb": "FLOAT",
                "postgres": "numeric",
                "flink": "FLOAT NOT NULL",
            },
            id="float32",
        ),
        param(
            ibis.literal(1, type=dt.float64),
            {
                "bigquery": "FLOAT64",
                "clickhouse": "Float64",
                "impala": "DECIMAL(2,1)",
                "snowflake": "INTEGER",
                "sqlite": "real",
                "trino": "double",
                "duckdb": "DOUBLE",
                "postgres": "numeric",
                "flink": "DOUBLE NOT NULL",
            },
            id="float64",
        ),
    ],
)
def test_numeric_literal(con, backend, expr, expected_types):
    result = con.execute(expr)
    assert result == 1

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == expected_types[backend_name]


@pytest.mark.parametrize(
    ("expr", "expected_result", "expected_types"),
    [
        param(
            ibis.literal(decimal.Decimal("1.1"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": decimal.Decimal("1.1"),
                "snowflake": "1.1",
                "sqlite": 1.1,
                "trino": 1.1,
                "dask": decimal.Decimal("1.1"),
                "duckdb": decimal.Decimal("1.1"),
                "postgres": 1.1,
                "pandas": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "mysql": 1.1,
                "mssql": 1.1,
                "druid": 1.1,
                "datafusion": decimal.Decimal("1.1"),
                "oracle": 1.1,
                "flink": decimal.Decimal("1.1"),
            },
            {
                "bigquery": "NUMERIC",
                "snowflake": "VARCHAR",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "flink": "DECIMAL(38, 18) NOT NULL",
            },
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ["impala"],
                    "impala.error.HiveServer2Error: AnalysisException: Syntax error in line 1:"
                    "SELECT typeof(Decimal('1.1')) AS `TypeOf(Decimal('1.1'))"
                    "Encountered: DECIMAL"
                    "Expected: ALL, CASE, CAST, DEFAULT, DISTINCT, EXISTS, FALSE, IF, "
                    "INTERVAL, LEFT, NOT, NULL, REPLACE, RIGHT, TRUNCATE, TRUE, IDENTIFIER"
                    "CAUSED BY: Exception: Syntax error",
                    raises=ImpalaHiveServer2Error,
                ),
            ],
            id="default",
        ),
        param(
            ibis.literal(decimal.Decimal("1.1"), type=dt.Decimal(38, 9)),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": decimal.Decimal("1.1"),
                "snowflake": "1.100000000",
                "sqlite": 1.1,
                "trino": 1.1,
                "duckdb": decimal.Decimal("1.100000000"),
                "postgres": 1.1,
                "pandas": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "mysql": 1.1,
                "clickhouse": decimal.Decimal("1.1"),
                "dask": decimal.Decimal("1.1"),
                "mssql": 1.1,
                "druid": 1.1,
                "datafusion": decimal.Decimal("1.1"),
                "oracle": 1.1,
                "flink": decimal.Decimal("1.1"),
            },
            {
                "bigquery": "NUMERIC",
                "clickhouse": "Decimal(38, 9)",
                "snowflake": "VARCHAR",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(38,9)",
                "postgres": "numeric",
                "flink": "DECIMAL(38, 9) NOT NULL",
            },
            marks=[
                pytest.mark.broken(
                    ["impala"],
                    "impala.error.HiveServer2Error: AnalysisException: Syntax error in line 1:"
                    "SELECT typeof(Decimal('1.1')) AS `TypeOf(Decimal('1.1'))"
                    "Encountered: DECIMAL"
                    "Expected: ALL, CASE, CAST, DEFAULT, DISTINCT, EXISTS, FALSE, IF, "
                    "INTERVAL, LEFT, NOT, NULL, REPLACE, RIGHT, TRUNCATE, TRUE, IDENTIFIER"
                    "CAUSED BY: Exception: Syntax error",
                    raises=ImpalaHiveServer2Error,
                ),
            ],
            id="decimal-small",
        ),
        param(
            ibis.literal(decimal.Decimal("1.1"), type=dt.Decimal(76, 38)),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": decimal.Decimal("1.1"),
                "snowflake": "1.10000000000000000000000000000000000000",
                "sqlite": 1.1,
                "trino": 1.1,
                "dask": decimal.Decimal("1.1"),
                "postgres": 1.1,
                "pandas": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "mysql": 1.1,
                "clickhouse": decimal.Decimal(
                    "1.10000000000000003193790845333396190208"
                ),
                "mssql": 1.1,
                "druid": 1.1,
                "oracle": 1.1,
            },
            {
                "bigquery": "BIGNUMERIC",
                "clickhouse": "Decimal(76, 38)",
                "snowflake": "VARCHAR",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
            },
            marks=[
                pytest.mark.broken(
                    ["impala"],
                    "impala.error.HiveServer2Error: AnalysisException: Syntax error in line 1:"
                    "SELECT typeof(Decimal('1.2')) AS `TypeOf(Decimal('1.2'))"
                    "Encountered: DECIMAL"
                    "Expected: ALL, CASE, CAST, DEFAULT, DISTINCT, EXISTS, FALSE, IF, "
                    "INTERVAL, LEFT, NOT, NULL, REPLACE, RIGHT, TRUNCATE, TRUE, IDENTIFIER"
                    "CAUSED BY: Exception: Syntax error",
                    raises=ImpalaHiveServer2Error,
                ),
                pytest.mark.broken(
                    ["duckdb"],
                    "(duckdb.ParserException) Parser Error: Width must be between 1 and 38!",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.notyet(
                    ["flink"],
                    "The precision can be up to 38 in Flink",
                    raises=ValueError,
                ),
            ],
            id="decimal-big",
        ),
        param(
            ibis.literal(decimal.Decimal("Infinity"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("inf"),
                "snowflake": "Infinity",
                "sqlite": float("inf"),
                "postgres": float("nan"),
                "pandas": decimal.Decimal("Infinity"),
                "dask": decimal.Decimal("Infinity"),
                "impala": float("inf"),
            },
            {
                "bigquery": "FLOAT64",
                "snowflake": "VARCHAR",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "impala": "DOUBLE",
            },
            marks=[
                pytest.mark.broken(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ["duckdb"],
                    "duckdb.ConversionException: Conversion Error: Could not cast value inf to DECIMAL(18,3)",
                    raises=DuckDBConversionException,
                ),
                pytest.mark.broken(
                    ["trino"],
                    "(trino.exceptions.TrinoUserError) TrinoUserError(type=USER_ERROR, name=INVALID_LITERAL, "
                    "message=\"line 1:51: 'Infinity' is not a valid decimal literal\", "
                    "query_id=20230128_024107_01084_y8zm3)",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["pyspark"],
                    "An error occurred while calling z:org.apache.spark.sql.functions.lit.",
                    raises=Py4JError,
                ),
                pytest.mark.broken(
                    ["mysql"],
                    "(pymysql.err.OperationalError) (1054, \"Unknown column 'Infinity' in 'field list'\")"
                    "[SQL: SELECT %(param_1)s AS `Decimal('Infinity')`]",
                    raises=sa.exc.OperationalError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    "(pymssql._pymssql.ProgrammingError) (207, b\"Invalid column name 'Infinity'."
                    "DB-Lib error message 20018, severity 16:\nGeneral SQL Server error: "
                    'Check messages from the SQL Server\n")'
                    "[SQL: SELECT %(param_1)s AS [Decimal('Infinity')]]",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 8 to line 1, "
                    "column 15: Column 'Infinity' not found in any table",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(["datafusion"], raises=Exception),
                pytest.mark.broken(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=sa.exc.DatabaseError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "Infinity is not supported in Flink SQL",
                    raises=ValueError,
                ),
            ],
            id="decimal-infinity+",
        ),
        param(
            ibis.literal(decimal.Decimal("-Infinity"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("-inf"),
                "snowflake": "-Infinity",
                "sqlite": float("-inf"),
                "postgres": float("nan"),
                "pandas": decimal.Decimal("-Infinity"),
                "dask": decimal.Decimal("-Infinity"),
                "impala": float("-inf"),
            },
            {
                "bigquery": "FLOAT64",
                "snowflake": "VARCHAR",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "impala": "DOUBLE",
            },
            marks=[
                pytest.mark.broken(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ["duckdb"],
                    "duckdb.ConversionException: Conversion Error: Could not cast value -inf to DECIMAL(18,3)",
                    raises=DuckDBConversionException,
                ),
                pytest.mark.broken(
                    ["trino"],
                    "(trino.exceptions.TrinoUserError) TrinoUserError(type=USER_ERROR, name=INVALID_LITERAL, "
                    "message=\"line 1:51: '-Infinity' is not a valid decimal literal\", "
                    "query_id=20230128_024107_01084_y8zm3)",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["pyspark"],
                    "An error occurred while calling z:org.apache.spark.sql.functions.lit.",
                    raises=Py4JError,
                ),
                pytest.mark.broken(
                    ["mysql"],
                    "(pymysql.err.OperationalError) (1054, \"Unknown column 'Infinity' in 'field list'\")"
                    "[SQL: SELECT %(param_1)s AS `Decimal('-Infinity')`]",
                    raises=sa.exc.OperationalError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    "(pymssql._pymssql.ProgrammingError) (207, b\"Invalid column name 'Infinity'."
                    "DB-Lib error message 20018, severity 16:\nGeneral SQL Server error: "
                    'Check messages from the SQL Server\n")'
                    "[SQL: SELECT %(param_1)s AS [Decimal('-Infinity')]]",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 9 to line 1, "
                    "column 16: Column 'Infinity' not found in any table",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(["datafusion"], raises=Exception),
                pytest.mark.broken(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=sa.exc.DatabaseError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "Infinity is not supported in Flink SQL",
                    raises=ValueError,
                ),
            ],
            id="decimal-infinity-",
        ),
        param(
            ibis.literal(decimal.Decimal("NaN"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("nan"),
                "snowflake": "NaN",
                "sqlite": None,
                "postgres": float("nan"),
                "pandas": decimal.Decimal("NaN"),
                "dask": decimal.Decimal("NaN"),
                "impala": float("nan"),
            },
            {
                "bigquery": "FLOAT64",
                "snowflake": "VARCHAR",
                "sqlite": "null",
                "trino": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "impala": "DOUBLE",
            },
            marks=[
                pytest.mark.broken(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ["duckdb"],
                    "(duckdb.InvalidInputException) Invalid Input Error: Attempting "
                    "to execute an unsuccessful or closed pending query result"
                    "Error: Invalid Input Error: Type DOUBLE with value nan can't be "
                    "cast because the value is out of range for the destination type INT64",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["trino"],
                    "(trino.exceptions.TrinoUserError) TrinoUserError(type=USER_ERROR, name=INVALID_LITERAL, "
                    "message=\"line 1:51: 'NaN' is not a valid decimal literal\", "
                    "query_id=20230128_024107_01084_y8zm3)",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["pyspark"],
                    "An error occurred while calling z:org.apache.spark.sql.functions.lit.",
                    raises=Py4JError,
                ),
                pytest.mark.broken(
                    ["mysql"],
                    "(pymysql.err.OperationalError) (1054, \"Unknown column 'NaN' in 'field list'\")"
                    "[SQL: SELECT %(param_1)s AS `Decimal('NaN')`]",
                    raises=sa.exc.OperationalError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    "(pymssql._pymssql.ProgrammingError) (207, b\"Invalid column name 'NaN'."
                    "DB-Lib error message 20018, severity 16:\nGeneral SQL Server error: "
                    'Check messages from the SQL Server\n")'
                    "[SQL: SELECT %(param_1)s AS [Decimal('NaN')]]",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 8 to line 1, column 10: Column 'NaN' not found in any table"
                    "[SQL: SELECT NaN AS \"Decimal('NaN')\"]",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 8 to line 1, "
                    "column 10: Column 'NaN' not found in any table",
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(["datafusion"], raises=Exception),
                pytest.mark.broken(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=sa.exc.DatabaseError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "NaN is not supported in Flink SQL",
                    raises=ValueError,
                ),
            ],
            id="decimal-NaN",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=ValueError)
def test_decimal_literal(con, backend, expr, expected_types, expected_result):
    backend_name = backend.name()

    result = con.execute(expr)
    current_expected_result = expected_result[backend_name]
    if type(current_expected_result) in (float, decimal.Decimal) and math.isnan(
        current_expected_result
    ):
        assert math.isnan(result) and type(result) == type(current_expected_result)
    else:
        assert result == current_expected_result

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == expected_types[backend_name]


@pytest.mark.parametrize(
    ("operand_fn", "expected_operand_fn"),
    [
        param(
            lambda t: t.float_col,
            lambda t: t.float_col,
            id="float-column",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="AttributeError: 'DecimalColumn' object has no attribute 'isinf'",
                )
            ],
        ),
        param(
            lambda t: t.double_col,
            lambda t: t.double_col,
            id="double-column",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="AttributeError: 'DecimalColumn' object has no attribute 'isinf'",
                )
            ],
        ),
        param(
            lambda t: ibis.literal(1.3),
            lambda t: 1.3,
            id="float-literal",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                )
            ],
        ),
        param(
            lambda t: ibis.literal(np.nan),
            lambda t: np.nan,
            id="nan-literal",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                )
            ],
        ),
        param(
            lambda t: ibis.literal(np.inf),
            lambda t: np.inf,
            id="inf-literal",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                )
            ],
        ),
        param(
            lambda t: ibis.literal(-np.inf),
            lambda t: -np.inf,
            id="-inf-literal",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("expr_fn", "expected_expr_fn"),
    [
        param(
            operator.methodcaller("isnan"),
            np.isnan,
            id="isnan",
        ),
        param(
            operator.methodcaller("isinf"),
            np.isinf,
            id="isinf",
        ),
    ],
)
@pytest.mark.notimpl(
    ["mysql", "sqlite", "datafusion", "mssql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.xfail(
    duckdb is not None and vparse(duckdb.__version__) < vparse("0.3.3"),
    reason="<0.3.3 does not support isnan/isinf properly",
)
def test_isnan_isinf(
    backend,
    con,
    alltypes,
    df,
    operand_fn,
    expected_operand_fn,
    expr_fn,
    expected_expr_fn,
):
    expr = expr_fn(operand_fn(alltypes)).name("tmp")
    expected = expected_expr_fn(expected_operand_fn(df))

    result = con.execute(expr)

    if isinstance(expected, pd.Series):
        expected = backend.default_series_rename(expected)
        backend.assert_series_equal(result, expected)
    else:
        try:
            assert result == expected
        except ValueError:
            backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L(-5).abs(), 5, id="abs-neg"),
        param(L(5).abs(), 5, id="abs"),
        param(
            ibis.least(L(10), L(1)),
            1,
            id="least",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
        param(
            ibis.greatest(L(10), L(1)),
            10,
            id="greatest",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
        param(
            L(5.5).round(),
            6.0,
            id="round",
        ),
        param(
            L(5.556).round(2),
            5.56,
            id="round-digits",
        ),
        param(L(5.556).ceil(), 6.0, id="ceil"),
        param(L(5.556).floor(), 5.0, id="floor"),
        param(
            L(5.556).exp(),
            math.exp(5.556),
            id="exp",
        ),
        param(
            L(5.556).sign(),
            1,
            id="sign-pos",
        ),
        param(
            L(-5.556).sign(),
            -1,
            id="sign-neg",
        ),
        param(
            L(0).sign(),
            0,
            id="sign-zero",
        ),
        param(L(5.556).sqrt(), math.sqrt(5.556), id="sqrt"),
        param(
            L(5.556).log(2),
            math.log(5.556, 2),
            id="log-base",
            marks=pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError),
        ),
        param(
            L(5.556).ln(),
            math.log(5.556),
            id="ln",
        ),
        param(
            L(5.556).log2(),
            math.log(5.556, 2),
            id="log2",
            marks=pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError),
        ),
        param(
            L(5.556).log10(),
            math.log10(5.556),
            id="log10",
        ),
        param(
            L(5.556).radians(),
            math.radians(5.556),
            id="radians",
        ),
        param(
            L(5.556).degrees(),
            math.degrees(5.556),
            id="degrees",
        ),
        param(
            L(11) % 3,
            11 % 3,
            id="mod",
        ),
    ],
)
def test_math_functions_literals(con, expr, expected):
    result = con.execute(expr.name("tmp"))
    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L(0.0).acos(), math.acos(0.0), id="acos"),
        param(L(0.0).asin(), math.asin(0.0), id="asin"),
        param(L(0.0).atan(), math.atan(0.0), id="atan"),
        param(L(0.0).atan2(1.0), math.atan2(0.0, 1.0), id="atan2"),
        param(L(0.0).cos(), math.cos(0.0), id="cos"),
        param(L(1.0).cot(), 1.0 / math.tan(1.0), id="cot"),
        param(L(0.0).sin(), math.sin(0.0), id="sin"),
        param(L(0.0).tan(), math.tan(0.0), id="tan"),
    ],
)
def test_trig_functions_literals(con, expr, expected):
    result = con.execute(expr.name("tmp"))
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    ("expr", "expected_fn"),
    [
        param(_.dc.acos(), np.arccos, id="acos"),
        param(_.dc.asin(), np.arcsin, id="asin"),
        param(_.dc.atan(), np.arctan, id="atan"),
        param(_.dc.atan2(_.dc), lambda c: np.arctan2(c, c), id="atan2"),
        param(_.dc.cos(), np.cos, id="cos"),
        param(_.dc.cot(), lambda c: 1.0 / np.tan(c), id="cot"),
        param(_.dc.sin(), np.sin, id="sin"),
        param(_.dc.tan(), np.tan, id="tan"),
    ],
)
def test_trig_functions_columns(backend, expr, alltypes, df, expected_fn):
    dc_max = df.double_col.max()
    expr = alltypes.mutate(dc=(_.double_col / dc_max).nullif(0)).select(tmp=expr)
    result = expr.tmp.to_pandas()
    expected = expected_fn((df.double_col / dc_max).replace(0.0, np.nan)).rename("tmp")
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t: (-t.double_col).abs(),
            lambda t: (-t.double_col).abs(),
            id="abs-neg",
        ),
        param(
            lambda t: t.double_col.abs(),
            lambda t: t.double_col.abs(),
            id="abs",
        ),
        param(
            lambda t: t.double_col.ceil(),
            lambda t: np.ceil(t.double_col).astype("int64"),
            id="ceil",
        ),
        param(
            lambda t: t.double_col.floor(),
            lambda t: np.floor(t.double_col).astype("int64"),
            id="floor",
        ),
        param(
            lambda t: t.double_col.sign(),
            lambda t: np.sign(t.double_col),
            id="sign",
            marks=[
                pytest.mark.notyet(
                    ["oracle"],
                    raises=AssertionError,
                    reason="Oracle SIGN function has different behavior for BINARY_FLOAT vs NUMERIC",
                ),
            ],
        ),
        param(
            lambda t: (-t.double_col).sign(),
            lambda t: np.sign(-t.double_col),
            id="sign-negative",
            marks=[
                pytest.mark.notyet(
                    ["oracle"],
                    raises=AssertionError,
                    reason="Oracle SIGN function has different behavior for BINARY_FLOAT vs NUMERIC",
                ),
            ],
        ),
    ],
)
def test_simple_math_functions_columns(
    backend, con, alltypes, df, expr_fn, expected_fn
):
    expr = expr_fn(alltypes).name("tmp")
    expected = backend.default_series_rename(expected_fn(df))
    result = con.execute(expr)
    backend.assert_series_equal(result, expected)


# we add one to double_col in this test to make sure the common case works (no
# domain errors), and we test the backends' various failure modes in each
# backend's test suite


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t: t.double_col.add(1).sqrt(),
            lambda t: np.sqrt(t.double_col + 1),
            id="sqrt",
        ),
        param(
            lambda t: t.double_col.add(1).exp(),
            lambda t: np.exp(t.double_col + 1),
            id="exp",
        ),
        param(
            lambda t: t.double_col.add(1).log(2),
            lambda t: np.log2(t.double_col + 1),
            id="log2",
        ),
        param(
            lambda t: t.double_col.add(1).ln(),
            lambda t: np.log(t.double_col + 1),
            id="ln",
        ),
        param(
            lambda t: t.double_col.add(1).log10(),
            lambda t: np.log10(t.double_col + 1),
            id="log10",
        ),
        param(
            lambda t: (t.double_col + 1).log(
                ibis.greatest(
                    9_000,
                    t.bigint_col,
                )
            ),
            lambda t: (
                np.log(t.double_col + 1) / np.log(np.maximum(9_000, t.bigint_col))
            ),
            id="log_base_bigint",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(
                    ["datafusion"],
                    raises=ValueError,
                    reason="Base greatest(9000, t0.bigint_col) for logarithm not supported!",
                ),
                pytest.mark.notimpl(["polars"], raises=com.UnsupportedArgumentError),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["druid"],
    raises=TypeError,
    reason="loop of ufunc does not support argument 0 of type float which has no callable log2 method",
)
def test_complex_math_functions_columns(
    backend, con, alltypes, df, expr_fn, expected_fn
):
    expr = expr_fn(alltypes).name("tmp")
    expected = backend.default_series_rename(expected_fn(df))
    result = con.execute(expr)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda be, t: t.double_col.round(),
            lambda be, t: be.round(t.double_col),
            id="round",
            marks=[
                pytest.mark.notimpl(["mssql"], raises=AssertionError),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=TypeError,
                    reason="loop of ufunc does not support argument 0 of type float which has no callable rint method",
                ),
            ],
        ),
        param(
            lambda be, t: t.double_col.add(0.05).round(3),
            lambda be, t: be.round(t.double_col + 0.05, 3),
            id="round-with-param",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=TypeError,
                    reason="loop of ufunc does not support argument 0 of type float which has no callable rint method",
                ),
            ],
        ),
        param(
            lambda be, t: ibis.least(t.bigint_col, t.int_col),
            lambda be, t: pd.Series(list(map(min, t.bigint_col, t.int_col))),
            id="least-all-columns",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
        param(
            lambda be, t: ibis.least(t.bigint_col, t.int_col, -2),
            lambda be, t: pd.Series(
                list(map(min, t.bigint_col, t.int_col, [-2] * len(t)))
            ),
            id="least-scalar",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
        param(
            lambda be, t: ibis.greatest(t.bigint_col, t.int_col),
            lambda be, t: pd.Series(list(map(max, t.bigint_col, t.int_col))),
            id="greatest-all-columns",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
        param(
            lambda be, t: ibis.greatest(t.bigint_col, t.int_col, -2),
            lambda be, t: pd.Series(
                list(map(max, t.bigint_col, t.int_col, [-2] * len(t)))
            ),
            id="greatest-scalar",
            marks=pytest.mark.notimpl(
                ["datafusion"], raises=com.OperationNotDefinedError
            ),
        ),
    ],
)
def test_backend_specific_numerics(backend, con, df, alltypes, expr_fn, expected_fn):
    expr = expr_fn(backend, alltypes)
    result = backend.default_series_rename(con.execute(expr.name("tmp")))
    expected = backend.default_series_rename(expected_fn(backend, df))
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
    ],
    ids=lambda op: op.__name__,
)
def test_binary_arithmetic_operations(backend, alltypes, df, op):
    smallint_col = alltypes.smallint_col + 1  # make it nonzero
    smallint_series = df.smallint_col + 1

    expr = op(alltypes.double_col, smallint_col).name("tmp")

    result = expr.execute()
    expected = op(df.double_col, smallint_series)
    if op is operator.floordiv:
        # defined in ops.FloorDivide.output_type
        # -> returns int64 whereas pandas float64
        result = result.astype("float64")

    expected = backend.default_series_rename(expected.astype("float64"))
    backend.assert_series_equal(result, expected, check_exact=False)


def test_mod(backend, alltypes, df):
    expr = operator.mod(alltypes.smallint_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.smallint_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.notimpl(["mssql"], raises=sa.exc.OperationalError)
@pytest.mark.notyet(
    ["druid"], raises=AssertionError, reason="mod with floats is integer mod"
)
@pytest.mark.notyet(
    ["bigquery"],
    reason="bigquery doesn't support floating modulus",
    raises=GoogleBadRequest,
)
def test_floating_mod(backend, alltypes, df):
    expr = operator.mod(alltypes.double_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.double_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_exact=False)


@pytest.mark.parametrize(
    "column",
    [
        param(
            "tinyint_col",
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=(sa.exc.DatabaseError, sa.exc.ArgumentError),
                    reason="Oracle doesn't do integer division by zero",
                )
            ],
        ),
        param(
            "smallint_col",
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=(sa.exc.DatabaseError, sa.exc.ArgumentError),
                    reason="Oracle doesn't do integer division by zero",
                )
            ],
        ),
        param(
            "int_col",
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=(sa.exc.DatabaseError, sa.exc.ArgumentError),
                    reason="Oracle doesn't do integer division by zero",
                )
            ],
        ),
        param(
            "bigint_col",
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=(sa.exc.DatabaseError, sa.exc.ArgumentError),
                    reason="Oracle doesn't do integer division by zero",
                )
            ],
        ),
        param(
            "float_col", marks=pytest.mark.notimpl(["druid"], raises=ZeroDivisionError)
        ),
        param(
            "double_col", marks=pytest.mark.notimpl(["druid"], raises=ZeroDivisionError)
        ),
    ],
)
@pytest.mark.notyet(["mysql", "pyspark"], raises=AssertionError)
@pytest.mark.notyet(
    ["duckdb", "sqlite"],
    raises=AssertionError,
    reason="returns NULL when dividing by zero",
)
@pytest.mark.notyet(["mssql"], raises=sa.exc.OperationalError)
@pytest.mark.notyet(["postgres"], raises=sa.exc.DataError)
@pytest.mark.notyet(["snowflake"], raises=sa.exc.ProgrammingError)
@pytest.mark.parametrize("denominator", [0, 0.0])
def test_divide_by_zero(backend, alltypes, df, column, denominator):
    expr = alltypes[column] / denominator
    result = expr.name("tmp").execute()

    expected = df[column].div(denominator)
    expected = backend.default_series_rename(expected).astype("float64")

    backend.assert_series_equal(result.astype("float64"), expected)


@pytest.mark.parametrize(
    ("default_precisions", "default_scales"),
    [
        (
            {
                "postgres": None,
                "mysql": 10,
                "snowflake": 38,
                "trino": 18,
                "duckdb": None,
                "sqlite": None,
                "mssql": None,
                "oracle": 38,
            },
            {
                "postgres": None,
                "mysql": 0,
                "snowflake": 0,
                "trino": 3,
                "duckdb": None,
                "sqlite": None,
                "mssql": None,
                "oracle": 0,
            },
        )
    ],
)
@pytest.mark.never(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "pandas",
        "pyspark",
        "polars",
    ],
    reason="Not SQLAlchemy backends",
)
@pytest.mark.notimpl(["druid"], raises=KeyError)
def test_sa_default_numeric_precision_and_scale(
    con, backend, default_precisions, default_scales, temp_table
):
    sa = pytest.importorskip("sqlalchemy")

    default_precision = default_precisions[backend.name()]
    default_scale = default_scales[backend.name()]

    typespec = [
        # name, sqlalchemy type, ibis type
        ("n1", sa.NUMERIC, dt.Decimal(default_precision, default_scale)),
        ("n2", sa.NUMERIC(5), dt.Decimal(5, default_scale)),
        ("n3", sa.NUMERIC(None, 4), dt.Decimal(default_precision, 4)),
        ("n4", sa.NUMERIC(10, 2), dt.Decimal(10, 2)),
    ]

    sqla_types = []
    ibis_types = []
    for name, t, ibis_type in typespec:
        sqla_types.append(sa.Column(name, t, nullable=True))
        ibis_types.append((name, ibis_type(nullable=True)))

    table = sa.Table(temp_table, sa.MetaData(), *sqla_types, quote=True)
    with con.begin() as bind:
        table.create(bind=bind, checkfirst=True)

    # Check that we can correctly recover the default precision and scale.
    schema = con._schema_from_sqla_table(table)
    expected = ibis.schema(ibis_types)

    assert_equal(schema, expected)


@pytest.mark.notimpl(
    ["dask", "pandas", "polars", "druid"], raises=com.OperationNotDefinedError
)
def test_random(con):
    expr = ibis.random()
    result = con.execute(expr)
    assert isinstance(result, float)
    assert 0 <= result <= 1


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        param(lambda x: x.clip(lower=0), lambda x: x.clip(lower=0), id="lower-int"),
        param(
            lambda x: x.clip(lower=0.0), lambda x: x.clip(lower=0.0), id="lower-float"
        ),
        param(lambda x: x.clip(upper=0), lambda x: x.clip(upper=0), id="upper-int"),
        param(
            lambda x: x.clip(lower=x - 1, upper=x + 1),
            lambda x: x.clip(lower=x - 1, upper=x + 1),
            marks=pytest.mark.notimpl(
                "polars",
                raises=com.UnsupportedArgumentError,
                reason="Polars does not support columnar argument Subtract(int_col, 1)",
            ),
            id="lower-upper-expr",
        ),
        param(
            lambda x: x.clip(lower=0, upper=1),
            lambda x: x.clip(lower=0, upper=1),
            id="lower-upper-int",
        ),
        param(
            lambda x: x.clip(lower=0, upper=1.0),
            lambda x: x.clip(lower=0, upper=1.0),
            id="lower-upper-float",
        ),
        param(
            lambda x: x.nullif(1).clip(lower=0),
            lambda x: x.where(x != 1).clip(lower=0),
            id="null-lower",
        ),
        param(
            lambda x: x.nullif(1).clip(upper=0),
            lambda x: x.where(x != 1).clip(upper=0),
            id="null-upper",
        ),
    ],
)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_clip(backend, alltypes, df, ibis_func, pandas_func):
    result = ibis_func(alltypes.int_col).execute()
    expected = pandas_func(df.int_col).astype(result.dtype)
    # Names won't match in the PySpark backend since PySpark
    # gives 'tmp' name when executing a Column
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason="SQL query requires 'MIN' operator that is not supported.",
)
def test_histogram(con, alltypes):
    n = 10
    hist = con.execute(alltypes.int_col.histogram(n).name("hist"))
    vc = hist.value_counts().sort_index()
    vc_np, _bin_edges = np.histogram(alltypes.int_col.execute(), bins=n)
    assert vc.tolist() == vc_np.tolist()


@pytest.mark.parametrize("const", ["pi", "e"])
def test_constants(con, const):
    expr = getattr(ibis, const)
    result = con.execute(expr)
    assert pytest.approx(result) == getattr(math, const)


pyspark_no_bitshift = pytest.mark.notyet(
    ["pyspark"],
    reason="pyspark doesn't implement bitshift operators",
    raises=com.OperationNotDefinedError,
)


@pytest.mark.parametrize("op", [and_, or_, xor])
@pytest.mark.parametrize(
    ("left_fn", "right_fn"),
    [
        param(lambda t: t.int_col, lambda t: t.int_col, id="col_col"),
        param(lambda _: 3, lambda t: t.int_col, id="scalar_col"),
        param(lambda t: t.int_col, lambda _: 3, id="col_scalar"),
    ],
)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
def test_bitwise_columns(backend, con, alltypes, df, op, left_fn, right_fn):
    expr = op(left_fn(alltypes), right_fn(alltypes)).name("tmp")
    result = con.execute(expr)

    expected = op(left_fn(df), right_fn(df)).rename("tmp")
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("op", "left_fn", "right_fn"),
    [
        param(
            lshift,
            lambda t: t.int_col,
            lambda t: t.int_col,
            id="lshift_col_col",
        ),
        param(
            lshift,
            lambda _: 3,
            lambda t: t.int_col,
            marks=pytest.mark.broken(
                ["impala"],
                reason="impala's behavior differs from every other backend",
                raises=AssertionError,
            ),
            id="lshift_scalar_col",
        ),
        param(lshift, lambda t: t.int_col, lambda _: 3, id="lshift_col_scalar"),
        param(rshift, lambda t: t.int_col, lambda t: t.int_col, id="rshift_col_col"),
        param(rshift, lambda _: 3, lambda t: t.int_col, id="rshift_scalar_col"),
        param(rshift, lambda t: t.int_col, lambda _: 3, id="rshift_col_scalar"),
    ],
)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
@pyspark_no_bitshift
def test_bitwise_shift(backend, alltypes, df, op, left_fn, right_fn):
    expr = op(left_fn(alltypes), right_fn(alltypes)).name("tmp")
    result = expr.execute()

    pandas_left = getattr(left := left_fn(df), "values", left)
    pandas_right = getattr(right := right_fn(df), "values", right)
    expected = pd.Series(
        op(pandas_left, pandas_right),
        name="tmp",
        dtype="int64",
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        param(and_),
        param(or_),
        param(xor),
        param(lshift, marks=pyspark_no_bitshift),
        param(rshift, marks=pyspark_no_bitshift),
    ],
)
@pytest.mark.parametrize(
    ("left", "right"),
    [param(4, L(2), id="int_col"), param(L(4), 2, id="col_int")],
)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
def test_bitwise_scalars(con, op, left, right):
    expr = op(left, right)
    result = con.execute(expr)
    expected = op(4, 2)
    assert result == expected


@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
def test_bitwise_not_scalar(con):
    expr = ~L(2)
    result = con.execute(expr)
    expected = -3
    assert result == expected


@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
def test_bitwise_not_col(backend, alltypes, df):
    expr = (~alltypes.int_col).name("tmp")
    result = expr.execute()
    expected = ~df.int_col
    backend.assert_series_equal(result, expected.rename("tmp"))
