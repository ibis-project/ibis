from __future__ import annotations

import contextlib
import decimal
import math
import operator
import sqlite3
from operator import and_, lshift, or_, rshift, xor

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis import _
from ibis import literal as L
from ibis.backends.tests.errors import (
    DatabricksServerOperationError,
    DuckDBParserException,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    OracleDatabaseError,
    PsycoPg2InternalError,
    PsycoPgDivisionByZero,
    Py4JError,
    Py4JJavaError,
    PyAthenaOperationalError,
    PyDruidProgrammingError,
    PyODBCDataError,
    PyODBCProgrammingError,
    PySparkArithmeticException,
    PySparkParseException,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.expr import datatypes as dt

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "athena": "integer",
                "duckdb": "INTEGER",
                "postgres": "integer",
                "risingwave": "integer",
                "flink": "INT NOT NULL",
                "databricks": "int",
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
                "trino": "real",
                "athena": "real",
                "duckdb": "DECIMAL(2,1)",
                "postgres": "numeric",
                "risingwave": "numeric",
                "flink": "DECIMAL(2, 1) NOT NULL",
                "databricks": "decimal(2,1)",
            },
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    "Unsupported type: Float16(nullable=True)",
                    raises=NotImplementedError,
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
                "trino": "real",
                "athena": "real",
                "duckdb": "DECIMAL(2,1)",
                "postgres": "numeric",
                "risingwave": "numeric",
                "flink": "DECIMAL(2, 1) NOT NULL",
                "databricks": "decimal(2,1)",
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
                "athena": "double",
                "duckdb": "DECIMAL(2,1)",
                "postgres": "numeric",
                "risingwave": "numeric",
                "flink": "DECIMAL(2, 1) NOT NULL",
                "databricks": "decimal(2,1)",
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
                "snowflake": decimal.Decimal("1.1"),
                "sqlite": decimal.Decimal("1.1"),
                "trino": decimal.Decimal("1.1"),
                "athena": decimal.Decimal(1),
                "exasol": decimal.Decimal(1),
                "duckdb": decimal.Decimal("1.1"),
                "impala": decimal.Decimal(1),
                "postgres": decimal.Decimal("1.1"),
                "risingwave": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "mysql": decimal.Decimal(1),
                "mssql": decimal.Decimal(1),
                "druid": decimal.Decimal("1.1"),
                "datafusion": decimal.Decimal("1.1"),
                "oracle": decimal.Decimal("1.1"),
                "flink": decimal.Decimal("1.1"),
                "polars": decimal.Decimal("1.1"),
                "databricks": decimal.Decimal("1.1"),
            },
            {
                "bigquery": "NUMERIC",
                "snowflake": "DECIMAL",
                "exasol": "DECIMAL(18,0)",
                "sqlite": "real",
                "impala": "DECIMAL(9,0)",
                "trino": "decimal(18,3)",
                "athena": "decimal(38,0)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "risingwave": "numeric",
                "flink": "DECIMAL(38, 18) NOT NULL",
                "databricks": "decimal(38,18)",
            },
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"],
                    reason="precision must be specified; clickhouse doesn't have a default",
                    raises=NotImplementedError,
                ),
            ],
            id="default",
        ),
        param(
            ibis.literal(decimal.Decimal("1.1"), type=dt.Decimal(38, 9)),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": decimal.Decimal("1.1"),
                "snowflake": decimal.Decimal("1.1"),
                "sqlite": decimal.Decimal("1.1"),
                "trino": decimal.Decimal("1.1"),
                "athena": decimal.Decimal("1.1"),
                "duckdb": decimal.Decimal("1.100000000"),
                "impala": decimal.Decimal("1.1"),
                "postgres": decimal.Decimal("1.1"),
                "risingwave": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "mysql": decimal.Decimal("1.1"),
                "clickhouse": decimal.Decimal("1.1"),
                "mssql": decimal.Decimal("1.1"),
                "druid": decimal.Decimal("1.1"),
                "datafusion": decimal.Decimal("1.1"),
                "oracle": decimal.Decimal("1.1"),
                "flink": decimal.Decimal("1.1"),
                "polars": decimal.Decimal("1.1"),
                "databricks": decimal.Decimal("1.1"),
            },
            {
                "bigquery": "NUMERIC",
                "clickhouse": "Decimal(38, 9)",
                "snowflake": "DECIMAL",
                "impala": "DECIMAL(38,9)",
                "sqlite": "real",
                "trino": "decimal(38,9)",
                "athena": "decimal(38,9)",
                "duckdb": "DECIMAL(38,9)",
                "postgres": "numeric",
                "risingwave": "numeric",
                "flink": "DECIMAL(38, 9) NOT NULL",
                "databricks": "decimal(38,9)",
            },
            marks=[pytest.mark.notimpl(["exasol"], raises=ExaQueryError)],
            id="decimal-small",
        ),
        param(
            ibis.literal(decimal.Decimal("1.1"), type=dt.Decimal(76, 38)),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": decimal.Decimal("1.1"),
                "sqlite": decimal.Decimal("1.1"),
                "postgres": decimal.Decimal("1.1"),
                "risingwave": decimal.Decimal("1.1"),
                "pyspark": decimal.Decimal("1.1"),
                "clickhouse": decimal.Decimal(
                    "1.10000000000000003193790845333396190208"
                ),
                "oracle": 1.1,
            },
            {
                "bigquery": "BIGNUMERIC",
                "clickhouse": "Decimal(76, 38)",
                "sqlite": "real",
                "trino": "decimal(2,1)",
                "athena": "decimal(2,1)",
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
                "risingwave": "numeric",
            },
            marks=[
                pytest.mark.notimpl(["exasol"], raises=ExaQueryError),
                pytest.mark.notimpl(["mysql"], raises=MySQLOperationalError),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["oracle"], raises=OracleDatabaseError),
                pytest.mark.notyet(["impala"], raises=ImpalaHiveServer2Error),
                pytest.mark.notyet(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notyet(
                    ["duckdb"],
                    reason="Unsupported precision.",
                    raises=DuckDBParserException,
                ),
                pytest.mark.notyet(
                    ["pyspark"],
                    reason="Unsupported precision.",
                    raises=(PySparkParseException, PySparkArithmeticException),
                ),
                pytest.mark.notyet(
                    ["trino"], reason="Unsupported precision.", raises=TrinoUserError
                ),
                pytest.mark.notyet(
                    ["athena"],
                    reason="Unsupported precision.",
                    raises=PyAthenaOperationalError,
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.notyet(
                    ["flink"],
                    "The precision can be up to 38 in Flink",
                    raises=Py4JJavaError,
                ),
                pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(["polars"], raises=TypeError),
                pytest.mark.notyet(
                    ["databricks"],
                    reason="Unsupported precision.",
                    raises=DatabricksServerOperationError,
                ),
            ],
            id="decimal-big",
        ),
        param(
            ibis.literal(decimal.Decimal("Infinity"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("inf"),
                "sqlite": decimal.Decimal("Infinity"),
                "postgres": decimal.Decimal("Infinity"),
                "risingwave": decimal.Decimal("Infinity"),
                "pyspark": decimal.Decimal("Infinity"),
                "exasol": float("inf"),
                "duckdb": float("inf"),
                "databricks": decimal.Decimal("Infinity"),
            },
            {
                "sqlite": "real",
                "postgres": "numeric",
                "risingwave": "numeric",
                "duckdb": "FLOAT",
                "databricks": "double",
            },
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 8 to line 1, "
                    "column 15: Column 'Infinity' not found in any table",
                    raises=PyDruidProgrammingError,
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.notyet(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=OracleDatabaseError,
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=TrinoUserError,
                    reason="can't cast infinity to decimal",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=PyAthenaOperationalError,
                    reason="can't cast infinity to decimal",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "Infinity is not supported in Flink SQL",
                    raises=Py4JJavaError,
                ),
                pytest.mark.notyet(
                    ["snowflake"],
                    "infinity is not allowed as a decimal value",
                    raises=SnowflakeProgrammingError,
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notyet(["polars"], reason="panic", raises=BaseException),
            ],
            id="decimal-infinity+",
        ),
        param(
            ibis.literal(decimal.Decimal("-Infinity"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("-inf"),
                "sqlite": decimal.Decimal("-Infinity"),
                "postgres": decimal.Decimal("-Infinity"),
                "risingwave": decimal.Decimal("-Infinity"),
                "pyspark": decimal.Decimal("-Infinity"),
                "exasol": float("-inf"),
                "duckdb": float("-inf"),
                "databricks": decimal.Decimal("-Infinity"),
            },
            {
                "sqlite": "real",
                "postgres": "numeric",
                "risingwave": "numeric",
                "duckdb": "FLOAT",
                "databricks": "double",
            },
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 9 to line 1, "
                    "column 16: Column 'Infinity' not found in any table",
                    raises=PyDruidProgrammingError,
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.notyet(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=OracleDatabaseError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "Infinity is not supported in Flink SQL",
                    raises=Py4JJavaError,
                ),
                pytest.mark.notyet(
                    ["snowflake"],
                    "infinity is not allowed as a decimal value",
                    raises=SnowflakeProgrammingError,
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=TrinoUserError,
                    reason="can't cast infinity to decimal",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=PyAthenaOperationalError,
                    reason="can't cast infinity to decimal",
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notyet(["polars"], reason="panic", raises=BaseException),
            ],
            id="decimal-infinity-",
        ),
        param(
            ibis.literal(decimal.Decimal("NaN"), type=dt.decimal),
            # TODO(krzysztof-kwitt): Should we unify it?
            {
                "bigquery": float("nan"),
                "snowflake": float("nan"),
                "sqlite": None,
                "postgres": float("nan"),
                "risingwave": float("nan"),
                "pyspark": decimal.Decimal("NaN"),
                "exasol": float("nan"),
                "duckdb": float("nan"),
                "databricks": decimal.Decimal("NaN"),
            },
            {
                "bigquery": "FLOAT64",
                "snowflake": "DOUBLE",
                "sqlite": "null",
                "postgres": "numeric",
                "risingwave": "numeric",
                "duckdb": "FLOAT",
                "databricks": "double",
            },
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    "Unsupported precision. Supported values: [1 : 76]. Current value: None",
                    raises=NotImplementedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(
                    ["druid"],
                    "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
                    "(org.apache.calcite.tools.ValidationException): "
                    "org.apache.calcite.runtime.CalciteContextException: From line 1, column 8 to line 1, "
                    "column 10: Column 'NaN' not found in any table",
                    raises=PyDruidProgrammingError,
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.notyet(
                    ["oracle"],
                    "(oracledb.exceptions.DatabaseError) DPY-4004: invalid number",
                    raises=OracleDatabaseError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "NaN is not supported in Flink SQL",
                    raises=Py4JJavaError,
                ),
                pytest.mark.notyet(
                    ["snowflake"],
                    "NaN is not allowed as a decimal value",
                    raises=SnowflakeProgrammingError,
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=TrinoUserError,
                    reason="can't cast nan to decimal",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=PyAthenaOperationalError,
                    reason="can't cast nan to decimal",
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notyet(["polars"], reason="panic", raises=BaseException),
            ],
            id="decimal-NaN",
        ),
    ],
)
def test_decimal_literal(con, backend, expr, expected_types, expected_result):
    backend_name = backend.name()
    result = con.execute(expr)
    current_expected_result = expected_result[backend_name]
    if type(current_expected_result) in (float, decimal.Decimal) and math.isnan(
        current_expected_result
    ):
        assert math.isnan(result)
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
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t: t.double_col,
            lambda t: t.double_col,
            id="double-column",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda _: ibis.literal(1.3),
            lambda _: 1.3,
            id="float-literal",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(lambda _: ibis.literal(np.nan), lambda _: np.nan, id="nan-literal"),
        param(
            lambda _: ibis.literal(np.inf),
            lambda _: np.inf,
            id="inf-literal",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(
            lambda _: ibis.literal(-np.inf),
            lambda _: -np.inf,
            id="-inf-literal",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
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
            marks=pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
            id="isnan",
        ),
        param(
            operator.methodcaller("isinf"),
            np.isinf,
            id="isinf",
            marks=pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
        ),
    ],
)
@pytest.mark.notimpl(["sqlite", "mssql", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["flink"], raises=(com.OperationNotDefinedError, NotImplementedError)
)
@pytest.mark.notimpl(["mysql"], raises=(MySQLOperationalError, NotImplementedError))
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
        param(ibis.least(L(10), L(1)), 1, id="least"),
        param(ibis.greatest(L(10), L(1)), 10, id="greatest"),
        param(L(5.5).round(), 6.0, id="round"),
        param(L(5.556).round(2), 5.56, id="round-digits"),
        param(L(5.556).ceil(), 6.0, id="ceil"),
        param(L(5.556).floor(), 5.0, id="floor"),
        param(L(5.556).exp(), math.exp(5.556), id="exp"),
        param(L(5.556).sign(), 1, id="sign-pos"),
        param(L(-5.556).sign(), -1, id="sign-neg"),
        param(L(0).sign(), 0, id="sign-zero"),
        param(L(5.556).sqrt(), math.sqrt(5.556), id="sqrt"),
        param(
            L(5.556).log(2),
            math.log2(5.556),
            id="log-base",
            marks=[
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function log10(numeric, numeric) does not exist",
                ),
            ],
        ),
        param(
            L(5.556).ln(),
            math.log(5.556),
            id="ln",
        ),
        param(L(5.556).ln(), math.log(5.556), id="ln"),
        param(
            L(5.556).log2(),
            math.log2(5.556),
            id="log2",
            marks=[
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function log10(numeric, numeric) does not exist",
                ),
            ],
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
        param(L(5.556).log10(), math.log10(5.556), id="log10"),
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
        param(L(11) % 3, 11 % 3, id="mod"),
        param(L(5.556).log10(), math.log10(5.556), id="log10"),
        param(L(5.556).radians(), math.radians(5.556), id="radians"),
        param(L(5.556).degrees(), math.degrees(5.556), id="degrees"),
        param(L(11) % 3, 11 % 3, id="mod"),
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
        param(
            _.dc.atan2(_.dc),
            lambda c: np.arctan2(c, c),
            id="atan2",
            marks=[
                pytest.mark.notyet(
                    ["mssql", "exasol"], raises=(PyODBCProgrammingError, ExaQueryError)
                )
            ],
        ),
        param(_.dc.cos(), np.cos, id="cos"),
        param(_.dc.sin(), np.sin, id="sin"),
        param(_.dc.tan(), np.tan, id="tan"),
    ],
)
def test_trig_functions_columns(backend, expr, alltypes, df, expected_fn):
    dc_max = df.double_col.max()
    expr = alltypes.mutate(dc=_.double_col / dc_max).select(tmp=expr)
    result = expr.tmp.to_pandas()
    expected = expected_fn(df.double_col / dc_max).rename("tmp")
    backend.assert_series_equal(result, expected)


def test_cotangent(backend, alltypes, df):
    dc_max = df.double_col.max()
    expr = alltypes.select(tmp=(0.5 + _.double_col / dc_max).cot())
    result = expr.tmp.to_pandas()
    expected = 1.0 / np.tan(0.5 + df.double_col / dc_max).rename("tmp")
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


def test_floor_divide_precedence(con):
    # Check that we compile to 16 / (4 / 2) and not 16 / 4 / 2
    expr = ibis.literal(16) // (4 / ibis.literal(2))
    result = int(con.execute(expr))
    assert result == 8


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
            marks=[
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function log10(numeric, numeric) does not exist",
                ),
            ],
            id="log2-explicit",
        ),
        param(
            lambda t: t.double_col.add(1).log2(),
            lambda t: np.log2(t.double_col + 1),
            marks=[
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function log10(numeric, numeric) does not exist",
                ),
            ],
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
                pytest.mark.notimpl(["polars"], raises=com.UnsupportedArgumentError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function log10(numeric, numeric) does not exist",
                ),
            ],
        ),
    ],
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
            lambda _, t: t.double_col.round(),
            lambda be, t: be.round(t.double_col),
            id="round",
            marks=[
                pytest.mark.notimpl(["mssql"], raises=AssertionError),
                pytest.mark.notyet(
                    ["druid"],
                    raises=AssertionError,
                    reason="rounding works but behavior differs from pandas",
                ),
                pytest.mark.notyet(
                    ["polars"],
                    raises=AssertionError,
                    reason="rounding behavior is slightly different",
                ),
            ],
        ),
        param(
            lambda _, t: t.double_col.add(0.05).round(3),
            lambda be, t: be.round(t.double_col + 0.05, 3),
            id="round-with-param",
        ),
        param(
            lambda _, t: ibis.least(t.bigint_col, t.int_col),
            lambda _, t: pd.Series(list(map(min, t.bigint_col, t.int_col))),
            id="least-all-columns",
        ),
        param(
            lambda _, t: ibis.least(t.bigint_col, t.int_col, -2),
            lambda _, t: pd.Series(
                list(map(min, t.bigint_col, t.int_col, [-2] * len(t)))
            ),
            id="least-scalar",
        ),
        param(
            lambda _, t: ibis.greatest(t.bigint_col, t.int_col),
            lambda _, t: pd.Series(list(map(max, t.bigint_col, t.int_col))),
            id="greatest-all-columns",
        ),
        param(
            lambda _, t: ibis.greatest(t.bigint_col, t.int_col, -2),
            lambda _, t: pd.Series(
                list(map(max, t.bigint_col, t.int_col, [-2] * len(t)))
            ),
            id="greatest-scalar",
        ),
    ],
)
def test_backend_specific_numerics(backend, con, df, alltypes, expr_fn, expected_fn):
    expr = expr_fn(backend, alltypes)
    result = backend.default_series_rename(con.execute(expr.name("tmp")))
    expected = backend.default_series_rename(expected_fn(backend, df))
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize("opname", ["add", "sub", "mul", "truediv", "floordiv", "pow"])
def test_binary_arithmetic_operations(backend, alltypes, df, opname):
    op = getattr(operator, opname)
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


def test_integer_truediv(con):
    expr = 1 / ibis.literal(2)
    result = con.execute(expr)
    assert result == 0.5


def test_mod(backend, alltypes, df):
    expr = operator.mod(alltypes.smallint_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.smallint_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.notimpl(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notyet(
    ["bigquery"],
    reason="bigquery doesn't support floating modulus",
    raises=GoogleBadRequest,
)
@pytest.mark.notyet(
    ["flink"],
    "Cannot apply '%' to arguments of type '<DOUBLE> % <SMALLINT>'. Supported form(s): '<EXACT_NUMERIC> % <EXACT_NUMERIC>",
    raises=Py4JError,
)
def test_floating_mod(backend, alltypes, df):
    expr = operator.mod(alltypes.double_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.double_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_exact=False)


@pytest.mark.parametrize(
    ("column", "denominator"),
    [
        param(
            "tinyint_col",
            0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
            ],
        ),
        param(
            "smallint_col",
            0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
            ],
        ),
        param(
            "int_col",
            0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
            ],
        ),
        param(
            "bigint_col",
            0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
            ],
        ),
        param("float_col", 0),
        param("double_col", 0),
        param(
            "tinyint_col",
            0.0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero"),
            ],
        ),
        param(
            "smallint_col",
            0.0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero"),
            ],
        ),
        param(
            "int_col",
            0.0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero"),
            ],
        ),
        param(
            "bigint_col",
            0.0,
            marks=[
                pytest.mark.notyet(
                    "oracle",
                    raises=OracleDatabaseError,
                    reason="Oracle doesn't do integer division by zero",
                ),
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero"),
            ],
        ),
        param(
            "float_col",
            0.0,
            marks=[
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero")
            ],
        ),
        param(
            "double_col",
            0.0,
            marks=[
                pytest.mark.never(["impala"], reason="doesn't allow divide by zero"),
            ],
        ),
    ],
)
@pytest.mark.notyet(["mysql", "pyspark"], raises=AssertionError)
@pytest.mark.notyet(["databricks"], raises=AssertionError, reason="returns NaNs")
@pytest.mark.notyet(
    ["sqlite"], raises=AssertionError, reason="returns NULL when dividing by zero"
)
@pytest.mark.notyet(["mssql"], raises=PyODBCDataError)
@pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError)
@pytest.mark.notyet(["postgres"], raises=PsycoPgDivisionByZero)
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.xfail_version(duckdb=["duckdb<1.1"])
def test_divide_by_zero(backend, alltypes, df, column, denominator):
    expr = alltypes[column] / denominator
    result = expr.name("tmp").execute()

    expected = df[column].div(denominator)
    expected = backend.default_series_rename(expected).astype("float64")

    backend.assert_series_equal(result.astype("float64"), expected)


@pytest.mark.notimpl(
    ["polars", "druid", "risingwave"], raises=com.OperationNotDefinedError
)
def test_random(con):
    expr = ibis.random()
    result = con.execute(expr)
    assert isinstance(result, float)
    assert 0 <= result <= 1


@pytest.mark.notimpl(
    ["polars", "druid", "risingwave"], raises=com.OperationNotDefinedError
)
def test_random_different_per_row(alltypes):
    result = alltypes.select("int_col", rand_col=ibis.random()).execute()
    assert result.rand_col.nunique() > 1


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
def test_clip(backend, alltypes, df, ibis_func, pandas_func):
    result = ibis_func(alltypes.int_col).execute()
    expected = pandas_func(df.int_col).astype(result.dtype)
    # Names won't match in the PySpark backend since PySpark
    # gives 'tmp' name when executing a Column
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="SQL query requires 'MIN' operator that is not supported.",
)
def test_histogram(con, alltypes):
    n = 10
    hist = con.execute(alltypes.int_col.histogram(nbins=n).name("hist"))
    vc = hist.value_counts().sort_index()
    vc_np, _bin_edges = np.histogram(alltypes.int_col.execute(), bins=n)
    expr = (
        ibis.memtable({"value": range(100)})
        .select(bin=_.value.histogram(nbins=10))
        .value_counts()
        .bin_count.nunique()
    )
    assert vc.tolist() == vc_np.tolist()
    assert con.execute(expr) == 1


@pytest.mark.parametrize("const", ["pi", "e"])
def test_constants(con, const):
    expr = getattr(ibis, const)
    result = con.execute(expr)
    assert pytest.approx(result) == getattr(math, const)


flink_no_bitwise = pytest.mark.notyet(
    ["flink"],
    reason="Flink doesn't implement bitwise operators",
    raises=Py4JError,
)


@pytest.mark.parametrize(
    "op",
    [
        and_,
        param(or_, marks=[pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)]),
        param(xor, marks=[pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)]),
    ],
)
@pytest.mark.parametrize(
    ("left_fn", "right_fn"),
    [
        param(lambda t: t.int_col, lambda t: t.int_col, id="col_col"),
        param(lambda _: 3, lambda t: t.int_col, id="scalar_col"),
        param(lambda t: t.int_col, lambda _: 3, id="col_scalar"),
    ],
)
@flink_no_bitwise
def test_bitwise_columns(backend, con, alltypes, df, op, left_fn, right_fn):
    expr = op(left_fn(alltypes), right_fn(alltypes)).name("tmp")
    result = con.execute(expr)

    expected = op(left_fn(df), right_fn(df)).rename("tmp")
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("op", "left_fn", "right_fn"),
    [
        param(lshift, lambda t: t.int_col, lambda t: t.int_col, id="lshift_col_col"),
        param(
            lshift,
            lambda _: 3,
            lambda t: t.int_col,
            marks=pytest.mark.notyet(
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
@pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError)
@pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)
@flink_no_bitwise
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
        and_,
        param(
            or_,
            marks=pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError),
        ),
        param(
            xor,
            marks=pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError),
        ),
        param(
            lshift,
            marks=[
                pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
            ],
        ),
        param(
            rshift,
            marks=[
                pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("left", "right"),
    [param(4, L(2), id="int_col"), param(L(4), 2, id="col_int")],
)
@flink_no_bitwise
def test_bitwise_scalars(con, op, left, right):
    expr = op(left, right)
    result = con.execute(expr)
    expected = op(4, 2)
    assert result == expected


@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)
@flink_no_bitwise
def test_bitwise_not_scalar(con):
    expr = ~L(2)
    result = con.execute(expr)
    expected = -3
    assert result == expected


@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)
@flink_no_bitwise
def test_bitwise_not_col(backend, alltypes, df):
    expr = (~alltypes.int_col).name("tmp")
    result = expr.execute()
    expected = ~df.int_col
    backend.assert_series_equal(result, expected.rename("tmp"))


def test_column_round_is_integer(con):
    t = ibis.memtable({"x": [1.2, 3.4]})
    expr = t.x.round().cast(int)
    result = con.execute(expr)

    one, three = sorted(result.tolist())

    assert one == 1
    assert isinstance(one, int)

    assert three == 3
    assert isinstance(three, int)


def test_scalar_round_is_integer(con):
    expr = ibis.literal(1.2).round().cast(int)
    result = con.execute(expr)

    assert result == 1
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "numbers",
    [
        param([1, 2, 3], id="ints"),
        param(
            [decimal.Decimal("1.1"), decimal.Decimal("2.2"), decimal.Decimal("3.3")],
            marks=[
                pytest.mark.notimpl(
                    ["sqlite"],
                    raises=(
                        sqlite3.ProgrammingError,
                        # With Python 3.10, the same code raises a different exception type :(
                        sqlite3.InterfaceError,
                    ),
                )
            ],
            id="decimals",
        ),
    ],
)
@pytest.mark.notyet(["exasol"], raises=ExaQueryError)
@pytest.mark.notimpl(["flink"], raises=NotImplementedError)
def test_memtable_decimal(con, numbers):
    schema = ibis.schema(dict(numbers=dt.Decimal(38, 9)))

    t = ibis.memtable({"numbers": numbers}, schema=schema)
    assert t.schema() == schema

    result = con.to_pyarrow(t)
    assert len(result) == len(numbers)
    assert result.schema == pa.schema({"numbers": pa.decimal128(38, 9)})
    assert result.schema == t.schema().to_pyarrow()
    assert result.schema == schema.to_pyarrow()
