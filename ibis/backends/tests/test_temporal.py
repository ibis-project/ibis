from __future__ import annotations

import contextlib
import datetime
import operator
import sqlite3
import sys
import warnings
from operator import methodcaller

import pytest
import sqlglot as sg
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends import _get_backend_names
from ibis.backends.tests.errors import (
    ArrowInvalid,
    ClickHouseDatabaseError,
    DuckDBInvalidInputException,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    ImpalaOperationalError,
    MySQLOperationalError,
    MySQLProgrammingError,
    OracleDatabaseError,
    PolarsInvalidOperationError,
    PolarsPanicException,
    PsycoPg2InternalError,
    Py4JJavaError,
    PyAthenaOperationalError,
    PyDruidProgrammingError,
    PyODBCDataError,
    PyODBCProgrammingError,
    PySparkConnectGrpcException,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.common.annotations import ValidationError
from ibis.conftest import IS_SPARK_REMOTE

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

sqlite_without_ymd_intervals = pytest.mark.notyet(
    ["sqlite"],
    condition=sqlite3.sqlite_version_info < (3, 46, 0),
    raises=com.UnsupportedOperationError,
)
sqlite_without_hms_intervals = pytest.mark.notyet(
    ["sqlite"],
    condition=sqlite3.sqlite_version_info < (3, 42, 0),
    raises=com.UnsupportedOperationError,
)


@pytest.mark.parametrize("attr", ["year", "month", "day"])
@pytest.mark.parametrize(
    "expr_fn",
    [
        param(
            methodcaller("date"),
            marks=[pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)],
            id="date",
        ),
        param(methodcaller("cast", "date"), id="cast"),
    ],
)
def test_date_extract(backend, alltypes, df, attr, expr_fn):
    expr = getattr(expr_fn(alltypes.timestamp_col), attr)()
    expected = getattr(df.timestamp_col.dt, attr).astype("int32")

    result = expr.name(attr).execute()

    backend.assert_series_equal(result, expected.rename(attr))


@pytest.mark.parametrize(
    "attr",
    [
        "year",
        "month",
        "day",
        param(
            "day_of_year",
            marks=[
                pytest.mark.notimpl(["impala"], raises=com.OperationNotDefinedError),
                pytest.mark.notyet(["oracle"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            "quarter",
            marks=[
                pytest.mark.notyet(["oracle"], raises=OracleDatabaseError),
            ],
        ),
        "hour",
        "minute",
        "second",
    ],
)
def test_timestamp_extract(backend, alltypes, df, attr):
    method = getattr(alltypes.timestamp_col, attr)
    expr = method().name(attr)
    result = expr.execute()
    expected = backend.default_series_rename(
        getattr(df.timestamp_col.dt, attr.replace("_", "")).astype("int32")
    ).rename(attr)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "transform", [toolz.identity, methodcaller("date")], ids=["timestamp", "date"]
)
@pytest.mark.notimpl(
    ["druid"],
    raises=(AttributeError, com.OperationNotDefinedError),
    reason="AttributeError: 'StringColumn' object has no attribute 'X'",
)
@pytest.mark.notyet(
    [
        "mysql",
        "sqlite",
        "mssql",
        "impala",
        "datafusion",
        "pyspark",
        "flink",
        "databricks",
    ],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't appear to support this operation directly",
)
def test_extract_iso_year(backend, alltypes, df, transform):
    value = transform(alltypes.timestamp_col)
    name = "iso_year"
    expr = value.iso_year().name(name)
    result = expr.execute()
    expected = backend.default_series_rename(
        df.timestamp_col.dt.isocalendar().year.astype("int32")
    ).rename(name)
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(
    ["druid"],
    raises=(AttributeError, com.OperationNotDefinedError),
    reason="AttributeError: 'StringColumn' object has no attribute 'X'",
)
@pytest.mark.notyet(
    [
        "mysql",
        "sqlite",
        "mssql",
        "impala",
        "datafusion",
        "pyspark",
        "flink",
        "databricks",
    ],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't appear to support this operation directly",
)
def test_iso_year_does_not_match_date_year(con):
    expr = ibis.date("2022-01-01").iso_year()
    assert con.execute(expr) == 2021


mark_notyet_risingwave_14670 = pytest.mark.notyet(
    ["risingwave"],
    raises=AssertionError,
    reason="Refer to https://github.com/risingwavelabs/risingwave/issues/14670",
)


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        param(methodcaller("year"), 2015, id="year"),
        param(methodcaller("month"), 9, id="month"),
        param(methodcaller("day"), 1, id="day"),
        param(methodcaller("hour"), 14, id="hour"),
        param(methodcaller("minute"), 48, id="minute"),
        param(methodcaller("second"), 5, id="second"),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
            marks=[
                pytest.mark.notimpl(
                    ["druid", "oracle"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda x: x.day_of_week.index(),
            1,
            id="day_of_week_index",
            marks=[
                pytest.mark.notimpl(
                    ["druid", "oracle", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda x: x.day_of_week.full_name(),
            "Tuesday",
            id="day_of_week_full_name",
            marks=[
                pytest.mark.notimpl(
                    ["druid", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
                mark_notyet_risingwave_14670,
            ],
        ),
    ],
)
def test_timestamp_extract_literal(con, func, expected):
    value = ibis.timestamp("2015-09-01 14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected


@pytest.mark.notimpl(["oracle", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["pyspark", "databricks"],
    raises=com.UnsupportedOperationError,
    reason="PySpark backend does not support extracting microseconds.",
)
@pytest.mark.notyet(
    ["impala"],
    raises=(ImpalaHiveServer2Error, ImpalaOperationalError),
    reason="Impala backend does not support extracting microseconds.",
)
@pytest.mark.notyet(["sqlite"], raises=AssertionError)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_timestamp_extract_microseconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.microsecond().name("microsecond")
    result = expr.execute()
    expected = backend.default_series_rename(
        (df.timestamp_col.dt.microsecond).astype("int32")
    ).rename("microsecond")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(["sqlite"], raises=AssertionError)
def test_timestamp_extract_milliseconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.millisecond().name("millisecond")
    result = expr.execute()
    expected = backend.default_series_rename(
        (df.timestamp_col.dt.microsecond // 1_000).astype("int32")
    ).rename("millisecond")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notimpl(
    ["bigquery"],
    raises=GoogleBadRequest,
    reason="UNIX_SECONDS does not support DATETIME arguments",
)
def test_timestamp_extract_epoch_seconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.epoch_seconds().name("tmp")
    result = expr.execute()

    expected = backend.default_series_rename(
        df.timestamp_col.astype("datetime64[ns]")
        .dt.floor("s")
        .astype("int64")
        .floordiv(1_000_000_000)
        .astype("int32")
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
def test_timestamp_extract_week_of_year(backend, alltypes, df):
    expr = alltypes.timestamp_col.week_of_year().name("tmp")
    result = expr.execute()
    expected = backend.default_series_rename(
        df.timestamp_col.dt.isocalendar().week.astype("int32")
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("ibis_unit", "pandas_unit"),
    [
        param("Y", "Y", id="year"),
        param("Q", "Q", id="quarter"),
        param("M", "M", id="month"),
        param(
            "W",
            "W",
            marks=[
                pytest.mark.notimpl(["mysql"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=AssertionError,
                    reason="implemented, but doesn't match other backends",
                ),
            ],
        ),
        param("D", "D"),
        param(
            "h",
            "h",
            marks=[
                pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError),
            ],
        ),
        param(
            "m",
            "min",
            marks=[
                pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError),
            ],
        ),
        param(
            "s",
            "s",
            marks=[
                pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError),
            ],
        ),
        param(
            "ms",
            "ms",
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "sqlite", "datafusion", "exasol"],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
            ],
        ),
        param(
            "us",
            "us",
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "sqlite", "trino", "datafusion", "exasol", "athena"],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="microseconds not supported in truncation",
                ),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
            ],
        ),
        param(
            "ns",
            "ns",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "duckdb",
                        "impala",
                        "mysql",
                        "postgres",
                        "risingwave",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "mssql",
                        "datafusion",
                        "exasol",
                        "druid",
                        "databricks",
                        "athena",
                    ],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=PolarsPanicException,
                    reason="attempt to calculate the remainder with a divisor of zero",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="nanoseconds not supported in truncation",
                ),
            ],
        ),
    ],
)
def test_timestamp_truncate(backend, alltypes, df, ibis_unit, pandas_unit):
    expr = alltypes.timestamp_col.truncate(ibis_unit).name("tmp")

    dtns = df.timestamp_col.dt

    if ibis_unit in ("Y", "Q", "M", "D", "W"):
        expected = dtns.to_period(pandas_unit).dt.to_timestamp()
    else:
        expected = dtns.floor(pandas_unit)

    result = expr.execute()
    expected = backend.default_series_rename(expected).astype(result.dtype)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "Q",
        "M",
        "D",
        param(
            "W",
            marks=[
                pytest.mark.notyet(["mysql"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=AssertionError,
                    reason="Implemented, but behavior doesn't match other backends",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
def test_date_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.date().truncate(unit).name("tmp")

    expected = df.timestamp_col.dt.to_period(unit).dt.to_timestamp().dt.date

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize(
    ("unit", "displacement_type"),
    [
        param(
            "Y",
            pd.offsets.DateOffset,
            # TODO - DateOffset - #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=TypeError,
                    reason="duration() got an unexpected keyword argument 'years'",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("Q", pd.offsets.DateOffset, marks=pytest.mark.xfail),
        param(
            "M",
            pd.offsets.DateOffset,
            # TODO - DateOffset - #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=TypeError,
                    reason="duration() got an unexpected keyword argument 'months'",
                ),
                pytest.mark.notyet(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-01839: date not valid for month specified",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            "W",
            pd.offsets.DateOffset,
            # TODO - DateOffset - #2553
            marks=[
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
                pytest.mark.notyet(
                    ["trino", "athena"],
                    raises=com.UnsupportedOperationError,
                    reason="week not implemented",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="ParseException: Encountered 'WEEK'. Was expecting one of: DAY, DAYS, HOUR",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Bind error: Invalid unit: week",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("D", pd.offsets.DateOffset, marks=sqlite_without_ymd_intervals),
        param("h", pd.Timedelta, marks=sqlite_without_hms_intervals),
        param("m", pd.Timedelta, marks=sqlite_without_hms_intervals),
        param("s", pd.Timedelta, marks=sqlite_without_hms_intervals),
        param(
            "ms",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="ParseException: Encountered 'MILLISECOND'. Was expecting one of: DAY, DAYS, HOUR, ...",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Bind error: Invalid unit: millisecond",
                ),
                sqlite_without_hms_intervals,
            ],
        ),
        param(
            "us",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse", "sqlite"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notimpl(
                    ["trino", "athena"],
                    raises=AssertionError,
                    reason="we're dropping microseconds to ensure results consistent with pandas",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="ParseException: Encountered 'MICROSECOND'. Was expecting one of: DAY, DAYS, HOUR, ...",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Bind error: Invalid unit: microsecond",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["druid", "exasol"], raises=com.OperationNotDefinedError)
def test_integer_to_interval_timestamp(
    backend, con, alltypes, df, unit, displacement_type
):
    interval = alltypes.int_col.as_interval(unit)
    expr = (alltypes.timestamp_col + interval).name("tmp")

    def convert_to_offset(offset, displacement_type=displacement_type):
        resolution = f"{interval.op().dtype.resolution}s"
        return displacement_type(**{resolution: offset})

    with warnings.catch_warnings():
        # both the implementation and test code raises pandas
        # PerformanceWarning, because We use DateOffset addition
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        result = con.execute(expr)
        offset = df.int_col.apply(convert_to_offset)
        expected = df.timestamp_col + offset

    expected = backend.default_series_rename(expected)
    expected = expected.astype(result.dtype)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        param(
            "Y",
            marks=[
                pytest.mark.notyet(
                    ["polars"], raises=TypeError, reason="not supported by polars"
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("Q", marks=pytest.mark.xfail),
        param(
            "M",
            marks=[
                pytest.mark.notyet(
                    ["polars"], raises=TypeError, reason="not supported by polars"
                ),
                pytest.mark.notyet(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-01839: date not valid for month specified",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            "W",
            marks=[
                pytest.mark.notyet(
                    ["trino", "athena"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Bind error: Invalid unit: week",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="week is not a valid unit in Flink",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("D", marks=sqlite_without_ymd_intervals),
    ],
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_integer_to_interval_date(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.as_interval(unit)
    month = alltypes.date_string_col[:2]
    day = alltypes.date_string_col[3:5]
    year = alltypes.date_string_col[6:8]
    date_col = ("20" + year + "-" + month + "-" + day).cast("date")
    expr = (date_col + interval).name("tmp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        result = con.execute(expr)

    def convert_to_offset(x):
        resolution = f"{interval.type().resolution}s"
        return pd.offsets.DateOffset(**{resolution: x})

    offset = df.int_col.apply(convert_to_offset)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=(UserWarning, pd.errors.PerformanceWarning)
        )
        expected = (
            pd.to_datetime(df.date_string_col).add(offset).astype("datetime64[s]")
        )

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


date_value = pd.Timestamp("2017-12-31")
timestamp_value = pd.Timestamp("2018-01-01 18:18:18")


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t, _: t.timestamp_col + ibis.interval(days=4),
            lambda t, _: t.timestamp_col + pd.Timedelta(days=4),
            id="timestamp-add-interval",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col
            + (ibis.interval(days=4) - ibis.interval(days=2)),
            lambda t, _: t.timestamp_col
            + (pd.Timedelta(days=4) - pd.Timedelta(days=2)),
            id="timestamp-add-interval-binop",
            marks=[
                pytest.mark.notimpl(
                    ["snowflake", "sqlite", "bigquery", "exasol", "mssql"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(["impala"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(["mysql"], raises=sg.ParseError),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=ValidationError,
                    reason="Given argument with datatype interval('D') is not implicitly castable to string",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col
            + (ibis.interval(days=4) + ibis.interval(hours=2)),
            lambda t, _: t.timestamp_col
            + (pd.Timedelta(days=4) + pd.Timedelta(hours=2)),
            id="timestamp-add-interval-binop-different-units",
            marks=[
                pytest.mark.notimpl(
                    ["sqlite", "polars", "snowflake", "bigquery", "exasol", "mssql"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(["impala"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(["mysql"], raises=sg.ParseError),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col - ibis.interval(days=17),
            lambda t, _: t.timestamp_col - pd.Timedelta(days=17),
            id="timestamp-subtract-interval",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col.date() + ibis.interval(days=4),
            lambda t, _: t.timestamp_col.dt.floor("d").add(pd.Timedelta(days=4)),
            id="date-add-interval",
            marks=[
                pytest.mark.notimpl(
                    ["exasol", "druid"], raises=com.OperationNotDefinedError
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col.date() - ibis.interval(days=14),
            lambda t, _: t.timestamp_col.dt.floor("d").sub(pd.Timedelta(days=14)),
            id="date-subtract-interval",
            marks=[
                pytest.mark.notimpl(
                    ["exasol", "druid"], raises=com.OperationNotDefinedError
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col - ibis.timestamp(timestamp_value),
            lambda t, _: pd.Series(
                t.timestamp_col.sub(timestamp_value).values.astype("timedelta64[s]")
            ).dt.floor("s"),
            id="timestamp-subtract-timestamp",
            marks=[
                pytest.mark.notimpl(
                    ["bigquery", "snowflake", "sqlite", "exasol", "mssql"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["athena"],
                    raises=PyAthenaOperationalError,
                    reason="not supported in hive",
                ),
                pytest.mark.notyet(
                    ["pyspark"],
                    condition=IS_SPARK_REMOTE,
                    raises=PySparkConnectGrpcException,
                    reason="arrow conversion breaks",
                ),
                pytest.mark.notyet(
                    ["databricks"],
                    raises=AssertionError,
                    reason="apparent over/underflow",
                ),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(
                    ["duckdb"],
                    raises=AssertionError,
                    reason="duckdb returns dateoffsets",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason=(
                        "CalciteContextException: Cannot apply '-' to arguments of type '<TIMESTAMP(9)> - <TIMESTAMP(0)>'."
                    ),
                ),
                pytest.mark.notimpl(
                    ["datafusion"],
                    raises=Exception,
                    reason="pyarrow.lib.ArrowInvalid: Casting from duration[us] to duration[s] would lose data",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param(
            lambda t, _: t.timestamp_col.date() - ibis.date(date_value),
            lambda t, _: pd.Series(
                (t.timestamp_col.dt.floor("d") - date_value).values.astype(
                    "timedelta64[D]"
                )
            ),
            id="date-subtract-date",
            marks=[
                pytest.mark.notimpl(
                    ["bigquery", "druid", "flink", "mssql"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["athena"],
                    raises=PyAthenaOperationalError,
                    reason="not supported in hive",
                ),
                pytest.mark.notyet(
                    ["datafusion"],
                    raises=Exception,
                    reason="pyarrow.lib.ArrowNotImplementedError: Unsupported cast",
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=com.OperationNotDefinedError,
                    reason="Some wonkiness in sqlglot generation.",
                ),
                pytest.mark.notyet(
                    ["pyspark"],
                    condition=IS_SPARK_REMOTE,
                    raises=PySparkConnectGrpcException,
                    reason="arrow conversion breaks",
                ),
                pytest.mark.notyet(
                    ["databricks"],
                    raises=AssertionError,
                    reason="apparent over/underflow",
                ),
            ],
        ),
    ],
)
def test_temporal_binop(backend, con, alltypes, df, expr_fn, expected_fn):
    expr = expr_fn(alltypes, backend).name("tmp")
    expected = expected_fn(df, backend)

    result = con.execute(expr)
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(
        result, expected.astype(result.dtype), check_dtype=False
    )


plus = lambda t, td: t.timestamp_col + pd.Timedelta(td)
minus = lambda t, td: t.timestamp_col - pd.Timedelta(td)


@pytest.mark.parametrize(
    ("timedelta", "temporal_fn"),
    [
        param(
            "36500d",
            plus,
            id="large-days-plus",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    # Note (mehmet): Following cannot be imported for backends other than Flink.
                    # raises=pyflink.util.exceptions.TableException,
                    raises=Exception,
                    reason="TableException: DAY_INTERVAL_TYPES precision is not supported: 5",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("5W", plus, id="weeks-plus", marks=sqlite_without_ymd_intervals),
        param("3d", plus, id="three-days-plus", marks=sqlite_without_ymd_intervals),
        param("2h", plus, id="two-hours-plus", marks=sqlite_without_hms_intervals),
        param("3m", plus, id="three-minutes-plus", marks=sqlite_without_hms_intervals),
        param("10s", plus, id="ten-seconds-plus", marks=sqlite_without_hms_intervals),
        param(
            "36500d",
            minus,
            id="large-days-minus",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    # Note (mehmet): Following cannot be imported for backends other than Flink.
                    # raises=pyflink.util.exceptions.TableException,
                    raises=Exception,
                    reason="TableException: DAY_INTERVAL_TYPES precision is not supported: 5",
                ),
                sqlite_without_ymd_intervals,
            ],
        ),
        param("5W", minus, id="weeks-minus", marks=sqlite_without_ymd_intervals),
        param("3d", minus, id="three-days-minus", marks=sqlite_without_ymd_intervals),
        param("2h", minus, id="two-hours-minus", marks=sqlite_without_hms_intervals),
        param(
            "3m", minus, id="three-minutes-minus", marks=sqlite_without_hms_intervals
        ),
        param("10s", minus, id="ten-seconds-minus", marks=sqlite_without_hms_intervals),
    ],
)
@pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_temporal_binop_pandas_timedelta(
    backend, con, alltypes, df, timedelta, temporal_fn
):
    expr = temporal_fn(alltypes, timedelta).name("tmp")
    expected = temporal_fn(df, timedelta)

    result = con.execute(expr)
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize("func_name", ["gt", "ge", "lt", "le", "eq", "ne"])
@pytest.mark.notimpl(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Invalid SQL; druid doesn't know about TIMESTAMPTZ",
)
def test_timestamp_comparison_filter(backend, con, alltypes, df, func_name):
    ts = pd.Timestamp("20100302", tz="UTC").to_pydatetime()

    comparison_fn = getattr(operator, func_name)
    expr = alltypes.filter(
        comparison_fn(alltypes.timestamp_col.cast("timestamp('UTC')"), ts)
    )

    col = df.timestamp_col.dt.tz_localize("UTC")
    expected = df[comparison_fn(col, ts)]
    result = con.execute(expr)

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func_name",
    ["gt", "ge", "lt", "le", "eq", "ne"],
)
@pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notimpl(
    ["polars"],
    raises=BaseException,  # pyo3_runtime.PanicException is not a subclass of Exception
    reason="failed to determine supertype of datetime[ns, UTC] and datetime[ns]",
)
@pytest.mark.never(
    ["bigquery"],
    raises=GoogleBadRequest,
    # perhaps we should consider disallowing this in ibis as well
    reason="BigQuery doesn't support comparing DATETIME and TIMESTAMP; numpy doesn't support timezones",
)
def test_timestamp_comparison_filter_numpy(backend, con, alltypes, df, func_name):
    ts = np.datetime64("2010-03-02 00:00:00.000123")

    comparison_fn = getattr(operator, func_name)
    expr = alltypes.filter(
        comparison_fn(alltypes.timestamp_col.cast("timestamp('UTC')"), ts)
    )

    ts = pd.Timestamp(ts.item(), tz="UTC")

    col = df.timestamp_col.dt.tz_localize("UTC")
    expected = df[comparison_fn(col, ts)]
    result = con.execute(expr)

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["exasol", "druid"], raises=com.OperationNotDefinedError)
@sqlite_without_ymd_intervals
def test_interval_add_cast_scalar(backend, alltypes):
    timestamp_date = alltypes.timestamp_col.date()
    delta = ibis.literal(10).cast("interval('D')")
    expr = (timestamp_date + delta).name("result")
    result = expr.execute()
    expected = timestamp_date.name("result").execute() + pd.Timedelta(10, unit="D")
    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.notimpl(["exasol", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["flink"], raises=AssertionError, reason="incorrect results")
@sqlite_without_ymd_intervals
def test_interval_add_cast_column(backend, alltypes, df):
    timestamp_date = alltypes.timestamp_col.date()
    delta = alltypes.bigint_col.cast("interval('D')")
    expr = alltypes.select("id", (timestamp_date + delta).name("tmp"))
    result = expr.execute().sort_values("id").reset_index().tmp

    df = df.sort_values("id").reset_index(drop=True)
    expected = (
        df["timestamp_col"]
        .dt.normalize()
        .add(df.bigint_col.astype("timedelta64[D]"))
        .rename("tmp")
    )

    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize(
    ("expr_fn", "pandas_pattern"),
    [
        param(
            lambda t: t.timestamp_col.strftime("%Y%m%d").name("formatted"),
            "%Y%m%d",
            id="literal_format_str",
        ),
        param(
            lambda t: (
                t.mutate(suffix=ibis.literal("%d"))
                .select(formatted=lambda t: t.timestamp_col.strftime("%Y%m" + t.suffix))
                .formatted
            ),
            "%Y%m%d",
            marks=[
                pytest.mark.notimpl(
                    [
                        "polars",
                    ],
                    raises=com.UnsupportedArgumentError,
                    reason="Polars does not support columnar argument StringConcat()",
                ),
                pytest.mark.notyet(["impala"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'StringColumn' object has no attribute 'strftime'",
                ),
            ],
            id="column_format_str",
        ),
    ],
)
@pytest.mark.notimpl(
    ["datafusion", "druid", "exasol"], raises=com.OperationNotDefinedError
)
def test_strftime(backend, alltypes, df, expr_fn, pandas_pattern):
    expr = expr_fn(alltypes)
    expected = df.timestamp_col.dt.strftime(pandas_pattern).rename("formatted")

    result = expr.execute()
    backend.assert_series_equal(result, expected)


unit_factors = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


@pytest.mark.parametrize(
    "unit",
    [
        "s",
        param(
            "ms",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark", "databricks"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit ms. Supported unit is s.",
                ),
                pytest.mark.notimpl(
                    ["clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`ms` unit is not supported!",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=AssertionError,
                    reason="athena or pyathena drops fractional seconds",
                ),
            ],
        ),
        param(
            "us",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark", "databricks"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit us. Supported unit is s.",
                ),
                pytest.mark.notimpl(["druid"], raises=com.UnsupportedArgumentError),
                pytest.mark.notimpl(
                    ["duckdb", "mssql", "clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`us` unit is not supported!",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=ValueError,
                    reason="<TimestampUnit.MICROSECOND: 'us'> unit is not supported!",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=AssertionError,
                    reason="athena or pyathena drops fractional seconds",
                ),
            ],
        ),
        param(
            "ns",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark", "databricks"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit ms. Supported unit is s.",
                ),
                pytest.mark.notimpl(["druid"], raises=com.UnsupportedArgumentError),
                pytest.mark.notimpl(
                    ["duckdb", "mssql", "clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`ns` unit is not supported!",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=ValueError,
                    reason="<TimestampUnit.MICROSECOND: 'us'> unit is not supported!",
                ),
                pytest.mark.notyet(
                    ["athena"],
                    raises=AssertionError,
                    reason="athena or pyathena drops fractional seconds",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["mysql", "postgres", "risingwave", "sqlite", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_integer_to_timestamp(backend, con, unit):
    backend_unit = backend.returned_timestamp_unit
    factor = unit_factors[unit]

    pandas_ts = pd.Timestamp("2018-04-13 09:54:11.872832").floor(unit).value

    # convert the timestamp to the input unit being tested
    int_expr = ibis.literal(pandas_ts // factor)
    expr_as = int_expr.as_timestamp(unit).name("tmp")
    result = con.execute(expr_as)

    expected = pd.Timestamp(pandas_ts, unit="ns").floor(backend_unit)

    assert result == expected


@pytest.mark.parametrize(
    "fmt",
    [
        # "11/01/10" - "month/day/year"
        param(
            "%m/%d/%y",
            id="mysql_format",
            marks=[
                pytest.mark.never(
                    ["snowflake"],
                    reason=(
                        "(snowflake.connector.errors.ProgrammingError) 100096 (22007): "
                        "Can't parse '11/01/10' as timestamp with format '%m/%d/%y'"
                    ),
                    raises=SnowflakeProgrammingError,
                ),
                pytest.mark.never(
                    ["flink"],
                    raises=ValueError,
                    reason="Datetime formatting style is not supported.",
                ),
            ],
        ),
        param(
            "MM/dd/yy",
            id="pyspark_format",
            marks=[
                pytest.mark.never(
                    ["bigquery"],
                    reason="400 Mismatch between format character 'M' and string character '0'",
                    raises=GoogleBadRequest,
                ),
                pytest.mark.never(
                    ["mysql"],
                    reason="NaTType does not support strftime",
                    raises=ValueError,
                ),
                pytest.mark.never(
                    ["trino"],
                    reason="datetime formatting style not supported",
                    raises=TrinoUserError,
                ),
                pytest.mark.never(
                    ["athena"],
                    reason="datetime formatting style not supported",
                    raises=PyAthenaOperationalError,
                ),
                pytest.mark.never(
                    ["polars"],
                    reason="datetime formatting style not supported",
                    raises=PolarsInvalidOperationError,
                ),
                pytest.mark.never(
                    ["duckdb"],
                    reason="datetime formatting style not supported",
                    raises=DuckDBInvalidInputException,
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["clickhouse", "sqlite", "datafusion", "mssql", "druid"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_string_as_timestamp(alltypes, fmt):
    table = alltypes
    result = table.mutate(date=table.date_string_col.as_timestamp(fmt)).execute()

    # TEST: do we get the same date out, that we put in?
    # format string assumes that we are using pandas' strftime
    for i, val in enumerate(result["date"]):
        assert val.strftime("%m/%d/%y") == result["date_string_col"][i]


@pytest.mark.parametrize(
    "fmt",
    [
        # "11/01/10" - "month/day/year"
        param(
            "%m/%d/%y",
            id="mysql_format",
            marks=pytest.mark.never(
                ["flink"],
                raises=ValueError,
                reason="Datetime formatting style is not supported.",
            ),
        ),
        param(
            "MM/dd/yy",
            id="pyspark_format",
            marks=[
                pytest.mark.never(
                    ["bigquery"],
                    reason="400 Mismatch between format character 'M' and string character '0'",
                    raises=GoogleBadRequest,
                ),
                pytest.mark.never(
                    ["mysql"],
                    reason="NaTType does not support strftime",
                    raises=ValueError,
                ),
                pytest.mark.never(
                    ["trino"],
                    reason="datetime formatting style not supported",
                    raises=TrinoUserError,
                ),
                pytest.mark.never(
                    ["athena"],
                    reason="datetime formatting style not supported",
                    raises=PyAthenaOperationalError,
                ),
                pytest.mark.never(
                    ["polars"],
                    reason="datetime formatting style not supported",
                    raises=PolarsInvalidOperationError,
                ),
                pytest.mark.never(
                    ["duckdb"],
                    reason="datetime formatting style not supported",
                    raises=DuckDBInvalidInputException,
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["clickhouse", "sqlite", "datafusion", "mssql", "druid"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
def test_string_as_date(alltypes, fmt):
    table = alltypes
    result = table.mutate(date=table.date_string_col.as_date(fmt)).execute()

    # TEST: do we get the same date out, that we put in?
    # format string assumes that we are using pandas' strftime
    for i, val in enumerate(result["date"]):
        assert val.strftime("%m/%d/%y") == result["date_string_col"][i]


@pytest.mark.notyet(
    [
        "pyspark",
        "exasol",
        "clickhouse",
        "impala",
        "mssql",
        "oracle",
        "trino",
        "druid",
        "datafusion",
        "flink",
        "databricks",
        "athena",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError)
def test_string_as_time(backend, alltypes):
    fmt = "%H:%M:%S"
    table = alltypes.mutate(
        time_string_col=alltypes.timestamp_col.truncate("s").time().cast(str)
    )
    expr = table.mutate(time=table.time_string_col.as_time(fmt))
    result = expr.execute()

    # TEST: do we get the same date out, that we put in?
    # format string assumes that we are using pandas' strftime
    backend.assert_series_equal(
        result["time"], result["timestamp_col"].dt.floor("s").dt.time.rename("time")
    )


@pytest.mark.parametrize(
    ("date", "expected_index", "expected_day"),
    [
        param("2017-01-01", 6, "Sunday", id="sunday"),
        param("2017-01-02", 0, "Monday", id="monday"),
        param("2017-01-03", 1, "Tuesday", id="tuesday"),
        param("2017-01-04", 2, "Wednesday", id="wednesday"),
        param("2017-01-05", 3, "Thursday", id="thursday"),
        param("2017-01-06", 4, "Friday", id="friday"),
        param("2017-01-07", 5, "Saturday", id="saturday"),
    ],
)
@pytest.mark.notimpl(["druid", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
@mark_notyet_risingwave_14670
def test_day_of_week_scalar(con, date, expected_index, expected_day):
    expr = ibis.literal(date).cast(dt.date)
    result_index = con.execute(expr.day_of_week.index().name("tmp"))
    assert result_index == expected_index

    result_day = con.execute(expr.day_of_week.full_name().name("tmp"))
    assert result_day.lower() == expected_day.lower()


@pytest.mark.notimpl(["oracle", "exasol", "druid"], raises=com.OperationNotDefinedError)
@mark_notyet_risingwave_14670
def test_day_of_week_column(backend, alltypes, df):
    expr = alltypes.timestamp_col.day_of_week

    result_index = expr.index().name("tmp").execute()
    expected_index = df.timestamp_col.dt.dayofweek.astype("int16")

    backend.assert_series_equal(result_index, expected_index, check_names=False)

    result_day = expr.full_name().name("tmp").execute()
    expected_day = df.timestamp_col.dt.day_name()

    backend.assert_series_equal(result_day, expected_day, check_names=False)


@pytest.mark.parametrize(
    ("day_of_week_expr", "day_of_week_pandas"),
    [
        param(
            lambda t: t.timestamp_col.day_of_week.index().count(),
            lambda s: s.dt.dayofweek.count(),
            id="day_of_week_index",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(
            lambda t: t.timestamp_col.day_of_week.full_name().length().sum(),
            lambda s: s.dt.day_name().str.len().sum(),
            id="day_of_week_full_name",
            marks=[mark_notyet_risingwave_14670],
        ),
    ],
)
@pytest.mark.notimpl(["oracle", "druid"], raises=com.OperationNotDefinedError)
def test_day_of_week_column_group_by(
    backend, alltypes, df, day_of_week_expr, day_of_week_pandas
):
    expr = alltypes.group_by("string_col").aggregate(
        day_of_week_result=day_of_week_expr
    )
    schema = expr.schema()
    assert schema["day_of_week_result"] == dt.int64

    result = expr.execute().sort_values("string_col")
    expected = (
        df.groupby("string_col")
        .timestamp_col.apply(day_of_week_pandas)
        .reset_index()
        .rename(columns={"timestamp_col": "day_of_week_result"})
    )

    backend.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.notimpl(["athena"], raises=PyAthenaOperationalError)
def test_now(con):
    expr = ibis.now()
    result = con.execute(expr.name("tmp"))
    assert isinstance(result, datetime.datetime)


@pytest.mark.notimpl(["athena"], raises=PyAthenaOperationalError)
def test_now_from_projection(alltypes):
    n = 2
    expr = alltypes.select(now=ibis.now()).limit(n)
    result = expr.execute()
    ts = result.now
    assert len(result) == n
    assert ts.nunique() == 1
    assert not pd.isna(ts.iat[0])


def test_today(con):
    result = con.execute(ibis.today())
    assert isinstance(result, datetime.date)


def test_today_from_projection(alltypes):
    expr = alltypes.select(today=ibis.today()).limit(2).today
    ts = expr.execute()
    assert len(ts) == 2
    assert ts.nunique() == 1
    years = expr.year().execute()
    assert years.nunique() == 1


DATE_BACKEND_TYPES = {
    "bigquery": "DATE",
    "clickhouse": "Date",
    "duckdb": "DATE",
    "flink": "DATE NOT NULL",
    "impala": "DATE",
    "postgres": "date",
    "snowflake": "DATE",
    "sqlite": "text",
    "trino": "date",
    "athena": "date",
    "risingwave": "date",
    "databricks": "date",
}


@pytest.mark.notimpl(["exasol", "druid"], raises=com.OperationNotDefinedError)
def test_date_literal(con, backend):
    expr = ibis.date(2022, 2, 4)
    result = con.execute(expr)
    assert result.strftime("%Y-%m-%d") == "2022-02-04"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == DATE_BACKEND_TYPES[backend_name]


TIMESTAMP_BACKEND_TYPES = {
    "bigquery": "DATETIME",
    "clickhouse": "DateTime",
    "impala": "TIMESTAMP",
    "snowflake": "TIMESTAMP_NTZ",
    "sqlite": "text",
    "trino": "timestamp(3)",
    "athena": "timestamp(3)",
    "duckdb": "TIMESTAMP",
    "postgres": "timestamp without time zone",
    "risingwave": "timestamp without time zone",
    "flink": "TIMESTAMP(6) NOT NULL",
    "databricks": "timestamp",
}


@pytest.mark.notimpl(
    ["pyspark", "mysql", "exasol", "oracle", "databricks"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(["impala"], raises=com.OperationNotDefinedError)
def test_timestamp_literal(con, backend):
    expr = ibis.timestamp(2022, 2, 4, 16, 20, 0)
    result = con.execute(expr)
    if not isinstance(result, str):
        result = result.strftime("%Y-%m-%d %H:%M:%S%Z")
    assert result == "2022-02-04 16:20:00"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == TIMESTAMP_BACKEND_TYPES[backend_name]


@pytest.mark.notimpl(
    ["mysql", "pyspark", "exasol", "databricks"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(["impala", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize(
    ("timezone", "expected"),
    [
        param("Europe/London", "2022-02-04 16:20:00GMT", id="name"),
        param(
            "PST8PDT",
            "2022-02-04 08:20:00PST",  # The time zone for Berkeley, California.
            id="iso",
            marks=[
                pytest.mark.notyet(
                    ["datafusion"],
                    raises=AssertionError,
                    reason="timezones don't seem to work",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["bigquery"],
    reason="timestamps with timezones other than 'UTC' not supported",
    raises=com.UnsupportedBackendType,
)
@pytest.mark.notimpl(
    ["sqlite"],
    reason="timestamps with timezones other than 'UTC' not supported",
    raises=com.UnsupportedOperationError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason=(
        "No match found for function signature make_timestamp(<NUMERIC>, <NUMERIC>, "
        "<NUMERIC>, <NUMERIC>, <NUMERIC>, <NUMERIC>)"
    ),
)
@pytest.mark.notimpl(["athena"], raises=PyAthenaOperationalError)
def test_timestamp_with_timezone_literal(con, timezone, expected):
    expr = ibis.timestamp(2022, 2, 4, 16, 20, 0).cast(dt.Timestamp(timezone=timezone))
    result = con.execute(expr)
    if not isinstance(result, str):
        result = result.strftime("%Y-%m-%d %H:%M:%S%Z")
    assert result == expected


TIME_BACKEND_TYPES = {
    "bigquery": "TIME",
    "flink": "TIME(0) NOT NULL",
    "snowflake": "TIME",
    "sqlite": "text",
    "trino": "time(3)",
    "athena": "time(3)",
    "duckdb": "TIME",
    "postgres": "time without time zone",
    "risingwave": "time without time zone",
}


@pytest.mark.notimpl(
    ["datafusion", "pyspark", "mysql", "oracle", "databricks"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["clickhouse", "impala", "exasol"], raises=com.OperationNotDefinedError
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError)
def test_time_literal(con, backend):
    expr = ibis.time(16, 20, 0)
    result = con.execute(expr)
    with contextlib.suppress(AttributeError):
        result = result.to_pytimedelta()
    assert str(result) == "16:20:00"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == TIME_BACKEND_TYPES[backend_name]


@pytest.mark.notyet(
    ["clickhouse", "impala"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't have a time datatype",
)
@pytest.mark.notyet(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notimpl(
    ["sqlite"], raises=AssertionError, reason="SQLite returns Timedelta from execution"
)
@pytest.mark.notyet(["oracle"], raises=OracleDatabaseError)
@pytest.mark.parametrize(
    "microsecond",
    [
        0,
        param(
            561021,
            marks=[
                pytest.mark.notimpl(
                    ["mysql"],
                    raises=AssertionError,
                    reason="doesn't have enough precision to capture microseconds",
                ),
                pytest.mark.notyet(["trino", "athena"], raises=AssertionError),
                pytest.mark.notyet(
                    ["flink"],
                    raises=AssertionError,
                    reason="flink doesn't preserve subsecond information",
                ),
            ],
        ),
    ],
    ids=["second", "subsecond"],
)
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notimpl(["athena"], raises=PyAthenaOperationalError)
@pytest.mark.notimpl(
    ["databricks"],
    raises=AssertionError,
    reason="returns a timedelta instead of a time",
)
def test_extract_time_from_timestamp(con, microsecond):
    raw_ts = datetime.datetime(2023, 1, 7, 13, 20, 5, microsecond)
    ts = ibis.timestamp(raw_ts)
    expr = ts.time()

    result = con.execute(expr)
    expected = raw_ts.time()

    assert result == expected


INTERVAL_BACKEND_TYPES = {
    "bigquery": "INTERVAL",
    "clickhouse": "IntervalSecond",
    "sqlite": "integer",
    "trino": "interval day to second",
    "athena": "interval day to second",
    "duckdb": "INTERVAL",
    "postgres": "interval",
    "risingwave": "interval",
}


@pytest.mark.notimpl(
    ["snowflake"],
    "(snowflake.connector.errors.ProgrammingError) 001007 (22023): SQL compilation error:"
    "invalid type [CAST(INTERVAL_LITERAL('second', '1') AS VARIANT)] for parameter 'TO_VARIANT'",
    raises=SnowflakeProgrammingError,
)
@pytest.mark.notyet(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notimpl(
    ["impala"],
    "AnalysisException: Syntax error in line 1: SELECT typeof(INTERVAL 1 SECOND) AS `TypeOf(1)` "
    "Encountered: ) Expected: +",
    raises=ImpalaHiveServer2Error,
)
@pytest.mark.notimpl(
    ["mysql"], "The backend implementation is broken. ", raises=MySQLProgrammingError
)
@pytest.mark.notimpl(
    ["bigquery", "duckdb"],
    reason="backend returns DateOffset arrays",
    raises=AssertionError,
)
@pytest.mark.notyet(
    ["datafusion"],
    raises=Exception,
    reason='This feature is not implemented: Can\'t create a scalar from array of type "Duration(Second)"',
)
@pytest.mark.notyet(
    ["clickhouse"],
    reason="Driver doesn't know how to handle intervals",
    raises=ClickHouseDatabaseError,
)
@pytest.mark.notimpl(
    ["flink"],
    raises=Py4JJavaError,
    reason=(
        "UnsupportedOperationException: Python vectorized UDF doesn't "
        "support logical type INTERVAL SECOND(3) NOT NULL currently"
    ),
)
@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notimpl(
    ["databricks"],
    reason="returns a different string format than expected in the test",
    raises=AssertionError,
)
@pytest.mark.notimpl(["athena"], raises=PyAthenaOperationalError)
def test_interval_literal(con, backend):
    expr = ibis.interval(1, unit="s")
    result = con.execute(expr)
    assert str(result) == "0 days 00:00:01"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == INTERVAL_BACKEND_TYPES[backend_name]


@pytest.mark.notimpl(["exasol", "druid"], raises=com.OperationNotDefinedError)
def test_date_column_from_ymd(backend, con, alltypes, df):
    c = alltypes.timestamp_col
    expr = ibis.date(c.year(), c.month(), c.day())
    tbl = alltypes.select(expr.name("timestamp_col"))
    result = con.execute(tbl)

    golden = df.timestamp_col.dt.date.astype(result.timestamp_col.dtype)
    backend.assert_series_equal(golden, result.timestamp_col)


@pytest.mark.notimpl(
    ["pyspark", "mysql", "exasol", "databricks"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(["impala", "oracle"], raises=com.OperationNotDefinedError)
def test_timestamp_column_from_ymdhms(backend, con, alltypes, df):
    c = alltypes.timestamp_col
    expr = ibis.timestamp(
        c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second()
    )
    tbl = alltypes.select(expr.name("timestamp_col"))
    result = con.execute(tbl)

    golden = df.timestamp_col.dt.floor("s").astype(result.timestamp_col.dtype)
    backend.assert_series_equal(golden, result.timestamp_col)


def test_date_scalar_from_iso(con):
    expr = ibis.literal("2022-02-24")
    expr2 = ibis.date(expr)

    result = con.execute(expr2)
    assert result.strftime("%Y-%m-%d") == "2022-02-24"


@pytest.mark.notimpl(["exasol"], raises=AssertionError, strict=False)
def test_date_column_from_iso(backend, con, alltypes, df):
    expr = (
        alltypes.year.cast("string")
        + "-"
        + alltypes.month.cast("string").lpad(2, "0")
        + "-13"
    )
    expr = ibis.date(expr)

    result = con.execute(expr.name("tmp"))
    golden = df.year.astype(str) + "-" + df.month.astype(str).str.rjust(2, "0") + "-13"
    actual = result.map(datetime.date.isoformat)
    backend.assert_series_equal(golden.rename("tmp"), actual.rename("tmp"))


@pytest.mark.notimpl(["druid", "oracle"], raises=com.OperationNotDefinedError)
def test_timestamp_extract_milliseconds_with_big_value(con):
    timestamp = ibis.timestamp("2021-01-01 01:30:59.333456")
    millis = timestamp.millisecond()
    result = con.execute(millis.name("tmp"))
    assert result == 333


@pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError, reason="ORA-00932")
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
def test_integer_cast_to_timestamp_column(backend, alltypes, df):
    expr = alltypes.int_col.cast("timestamp")
    expected = pd.to_datetime(df.int_col, unit="s").rename(expr.get_name())
    result = expr.execute()
    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError)
def test_integer_cast_to_timestamp_scalar(alltypes, df):
    expr = alltypes.int_col.min().cast("timestamp")
    result = expr.execute()
    expected = pd.to_datetime(df.int_col.min(), unit="s")
    assert result == expected


@pytest.mark.notimpl(
    ["clickhouse"],
    raises=com.UnsupportedOperationError,
    reason="Results in Timestamp('2023-11-04 14:47:18') (no subsecond) https://github.com/ClickHouse/ClickHouse/issues/29386",
)
@pytest.mark.notimpl(
    ["exasol"],
    raises=ExaQueryError,
    reason="conversion of float to timestamp is not supported",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.UnsupportedOperationError,
    reason="flink only supports integers as input to timestamp",
)
@pytest.mark.notimpl(
    ["impala"],
    raises=AssertionError,
    reason="result is Timestamp('2023-11-04 14:47:18.499999')",
)
@pytest.mark.notimpl(
    ["mssql"],
    raises=PyODBCDataError,
    reason="[22018] [FreeTDS][SQL Server]Explicit conversion from data type float to datetime2 is not allowed. (529)",
)
@pytest.mark.notimpl(
    ["oracle"],
    raises=OracleDatabaseError,
    reason="ORA-00932: expression is of data type NUMBER, which is incompatible with expected data type TIMESTAMP",
)
@pytest.mark.parametrize(
    "dtype", ["timestamp", "timestamp(1)", "timestamp(3)", "timestamp(6)"]
)
def test_subsecond_cast_to_timestamp(con, dtype):
    if con.name == "sqlite" and sys.platform == "win32" and sys.version_info >= (3, 9):
        pytest.skip("sqlite on Python 3.9 on Windows casts to NaT")
    expr = ibis.literal("1699109238.5").cast(float).cast(dtype)
    result = con.execute(expr)
    expected = pd.Timestamp("2023-11-04 14:47:18.5")
    assert expected == result


@pytest.mark.notimpl(
    ["clickhouse", "athena"],
    raises=AssertionError,
    reason="clickhouse truncates the result",
)
@pytest.mark.notimpl(["druid"], reason="timezone doesn't match", raises=AssertionError)
@pytest.mark.notyet(
    ["pyspark"],
    reason="PySpark doesn't handle big timestamps",
    condition=not IS_SPARK_REMOTE,
    raises=pd.errors.OutOfBoundsDatetime,
)
@pytest.mark.notyet(
    ["databricks"],
    reason="returns a value with a timezone, which the test doesn't expect",
    raises=AssertionError,
)
@pytest.mark.notimpl(["flink"], raises=ArrowInvalid)
@pytest.mark.notyet(["polars"], raises=PolarsInvalidOperationError)
def test_big_timestamp(con):
    # TODO: test with a timezone
    ts = "2419-10-11 10:10:25"
    value = ibis.timestamp(ts)
    result = con.execute(value.name("tmp"))
    expected = datetime.datetime.fromisoformat(ts)
    assert result == expected


DATE = datetime.date(2010, 11, 1)


def build_date_col(t):
    return (
        t.year.cast("string")
        + "-"
        + t.month.cast("string").lpad(2, "0")
        + "-"
        + (t.int_col + 1).cast("string").lpad(2, "0")
    ).cast("date")


@pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.parametrize(
    ("left_fn", "right_fn"),
    [
        param(build_date_col, lambda _: DATE, id="column_date"),
        param(lambda _: DATE, build_date_col, id="date_column"),
    ],
)
def test_timestamp_date_comparison(backend, alltypes, df, left_fn, right_fn):
    left = left_fn(alltypes)
    right = right_fn(alltypes)
    expr = left == right
    result = expr.name("result").execute()
    expected = (
        pd.to_datetime(
            (
                df.year.astype(str)
                .add("-")
                .add(df.month.astype(str).str.rjust(2, "0"))
                .add("-")
                .add(df.int_col.add(1).astype(str).str.rjust(2, "0"))
            ),
            format="%Y-%m-%d",
            exact=True,
        )
        .eq(pd.Timestamp(DATE))
        .rename("result")
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(
    ["clickhouse", "athena"], reason="returns incorrect results", raises=AssertionError
)
@pytest.mark.notimpl(
    ["pyspark"], condition=not IS_SPARK_REMOTE, raises=pd.errors.OutOfBoundsDatetime
)
@pytest.mark.notimpl(["polars"], raises=PolarsInvalidOperationError)
@pytest.mark.notyet(
    ["flink"],
    reason="Casting from timestamp[s] to timestamp[ns] would result in out of bounds timestamp: 81953424000",
    raises=ArrowInvalid,
)
def test_large_timestamp(con):
    huge_timestamp = datetime.datetime(year=4567, month=1, day=1)
    expr = ibis.timestamp("4567-01-01 00:00:00")
    result = con.execute(expr)
    assert result.replace(tzinfo=None) == huge_timestamp


@pytest.mark.parametrize(
    ("ts", "scale", "unit"),
    [
        param("2023-01-07 13:20:05.561", 3, "ms", id="ms"),
        param(
            "2023-01-07 13:20:05.561021",
            6,
            "us",
            id="us",
            marks=[
                pytest.mark.notyet(
                    ["sqlite", "flink"],
                    reason="doesn't support microseconds",
                    raises=AssertionError,
                ),
                pytest.mark.notyet(
                    ["druid"],
                    reason="time_parse truncates to milliseconds",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(["exasol"], raises=AssertionError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
            ],
        ),
        param(
            "2023-01-07 13:20:05.561000231",
            9,
            "ns",
            id="ns",
            marks=[
                pytest.mark.notyet(
                    ["impala", "pyspark", "trino", "databricks"],
                    reason="drivers appear to truncate nanos",
                    raises=AssertionError,
                ),
                pytest.mark.xfail_version(
                    duckdb=["duckdb<1.1"],
                    reason="not implemented until 1.1",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(
                    ["druid"],
                    reason="ibis normalization truncates nanos",
                    raises=AssertionError,
                ),
                pytest.mark.notyet(
                    ["postgres", "sqlite"],
                    reason="doesn't support nanoseconds",
                    raises=AssertionError,
                ),
                pytest.mark.notyet(
                    ["mssql"],
                    reason="doesn't support nanoseconds",
                    raises=PyODBCProgrammingError,
                ),
                pytest.mark.notyet(
                    ["mysql"],
                    reason="doesn't support nanoseconds",
                    raises=MySQLOperationalError,
                ),
                pytest.mark.notyet(
                    ["bigquery"],
                    reason=(
                        "doesn't support nanoseconds. "
                        "Server returns: 400 Invalid timestamp: '2023-01-07 13:20:05.561000231'"
                    ),
                    raises=GoogleBadRequest,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    reason="assert Timestamp('2023-01-07 13:20:05.561000') == Timestamp('2023-01-07 13:20:05.561000231')",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(["exasol"], raises=AssertionError),
                pytest.mark.notyet(
                    ["risingwave"],
                    raises=ValueError,
                    reason="Only supports up to microseconds",
                ),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["oracle"],
    raises=OracleDatabaseError,
    reason="ORA-01843: invalid month was specified",
)
def test_timestamp_precision_output(con, ts, scale, unit):
    dtype = dt.Timestamp(scale=scale)
    expr = ibis.literal(ts).cast(dtype)
    result = con.execute(expr)
    expected = pd.Timestamp(ts).floor(unit)
    assert result == expected


@pytest.mark.notimpl(["datafusion", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize(
    ("start", "end", "unit", "expected"),
    [
        param(
            ibis.time("01:58:00"),
            ibis.time("23:59:59"),
            "hour",
            22,
            id="time",
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"],
                    raises=com.OperationNotDefinedError,
                    reason="time types not yet implemented in ibis for the clickhouse backend",
                ),
                pytest.mark.notyet(
                    ["postgres", "risingwave"],
                    reason="postgres doesn't have any easy way to accurately compute the delta in specific units",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["exasol", "polars", "sqlite", "oracle", "impala"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(ibis.date("1992-09-30"), ibis.date("1992-10-01"), "day", 1, id="date"),
        param(
            ibis.timestamp("1992-09-30 23:59:59"),
            ibis.timestamp("1992-10-01 01:58:00"),
            "hour",
            2,
            id="timestamp",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark", "databricks"],
                    raises=AssertionError,
                    reason="backend computes timezone aware difference",
                ),
                pytest.mark.notimpl(
                    ["mysql"],
                    raises=com.OperationNotDefinedError,
                    reason="timestampdiff rounds after subtraction and mysql doesn't have a date_trunc function",
                ),
                pytest.mark.notimpl(
                    ["exasol", "polars", "sqlite", "oracle", "impala"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
    ],
)
def test_delta(con, start, end, unit, expected):
    expr = end.delta(start, unit=unit)
    assert con.execute(expr) == expected


@pytest.mark.notimpl(
    ["impala", "mysql", "pyspark", "sqlite", "trino", "druid", "databricks", "athena"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize(
    ("kws", "pd_freq"),
    [
        param(
            {"milliseconds": 50},
            "50ms",
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="backend doesn't support sub-second interval precision",
                ),
                pytest.mark.notimpl(
                    ["snowflake"],
                    raises=SnowflakeProgrammingError,
                    reason="snowflake doesn't support sub-second interval precision",
                ),
                pytest.mark.notimpl(
                    ["datafusion"],
                    raises=com.UnsupportedOperationError,
                    reason="backend doesn't support sub-second interval precision",
                ),
                pytest.mark.notimpl(
                    ["flink"],
                    raises=Py4JJavaError,
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=com.OperationNotDefinedError,
                    reason="TimestampBucket not implemented",
                ),
            ],
            id="milliseconds",
        ),
        param(
            {"seconds": 2},
            "2s",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="seconds",
        ),
        param(
            {"minutes": 5},
            "300s",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="minutes",
        ),
        param(
            {"hours": 2},
            "2h",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="hours",
        ),
        param(
            {"days": 2},
            "2D",
            marks=[
                pytest.mark.notimpl(
                    ["flink"],
                    raises=AssertionError,
                    reason="numpy array values are different (50.0 %)",
                ),
                pytest.mark.notimpl(
                    ["datafusion", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="days",
        ),
    ],
)
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function date_bin(interval, timestamp without time zone, timestamp without time zone) does not exist",
)
def test_timestamp_bucket(backend, kws, pd_freq):
    ts = backend.functional_alltypes.timestamp_col.execute().rename("ts")
    res = backend.functional_alltypes.timestamp_col.bucket(**kws).execute().rename("ts")
    sol = ts.dt.floor(pd_freq)
    backend.assert_series_equal(res, sol)


@pytest.mark.notimpl(
    [
        "datafusion",
        "impala",
        "mysql",
        "oracle",
        "pyspark",
        "sqlite",
        "trino",
        "druid",
        "databricks",
        "athena",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["clickhouse", "mssql", "snowflake"],
    reason="offset arg not supported",
    raises=com.UnsupportedOperationError,
)
@pytest.mark.parametrize("offset_mins", [2, -2], ids=["pos", "neg"])
@pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function date_bin(interval, timestamp without time zone, timestamp without time zone) does not exist",
)
def test_timestamp_bucket_offset(backend, offset_mins):
    ts = backend.functional_alltypes.timestamp_col
    expr = ts.bucket(minutes=5, offset=ibis.interval(minutes=offset_mins))
    res = expr.execute().astype("datetime64[ns]").rename("ts")
    td = pd.Timedelta(minutes=offset_mins)
    sol = ((ts.execute().rename("ts") - td).dt.floor("300s") + td).astype(
        "datetime64[ns]"
    )
    backend.assert_series_equal(res, sol)


_NO_SQLGLOT_DIALECT = ("flink", "polars")
no_sqlglot_dialect = sorted(
    param(backend, marks=pytest.mark.xfail) for backend in _NO_SQLGLOT_DIALECT
)


@pytest.mark.parametrize(
    "value",
    [
        param(datetime.date(2023, 4, 7), id="date"),
        param(datetime.datetime(2023, 4, 7, 4, 5, 6, 230136), id="timestamp"),
    ],
)
@pytest.mark.parametrize(
    "dialect",
    [*sorted(_get_backend_names(exclude=_NO_SQLGLOT_DIALECT)), *no_sqlglot_dialect],
)
def test_temporal_literal_sql(value, dialect, snapshot):
    expr = ibis.literal(value)
    sql = ibis.to_sql(expr, dialect=dialect)
    snapshot.assert_match(sql, "out.sql")


no_time_type = pytest.mark.xfail(
    raises=NotImplementedError, reason="no time type support"
)


@pytest.mark.parametrize(
    "dialect",
    [
        *sorted(
            _get_backend_names(
                exclude=(
                    "pyspark",
                    "impala",
                    "clickhouse",
                    "oracle",
                    *_NO_SQLGLOT_DIALECT,
                )
            )
        ),
        *no_sqlglot_dialect,
    ],
)
@pytest.mark.parametrize("micros", [0, 234567])
def test_time_literal_sql(dialect, snapshot, micros):
    value = datetime.time(4, 5, 6, microsecond=micros)
    expr = ibis.literal(value)
    sql = ibis.to_sql(expr, dialect=dialect)
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.notimpl(
    ["druid"], raises=PyDruidProgrammingError, reason="no date support"
)
@pytest.mark.parametrize(
    "value",
    [
        param("2017-12-31", id="simple"),
        param(
            "9999-01-02",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="clickhouse doesn't support dates after 2149-06-06",
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.xfail_version(
                    pyspark=["pyspark<3.5"],
                    raises=pd._libs.tslib.OutOfBoundsDatetime,
                    reason=(
                        "versions of pandas supported by PySpark <3.5 don't allow "
                        "pd.Timestamps with out-of-bounds timestamp values"
                    ),
                ),
            ],
            id="large",
        ),
        param(
            "0001-07-17",
            id="small",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="clickhouse doesn't support dates before the UNIX epoch",
                ),
                pytest.mark.notyet(["datafusion"], raises=Exception),
                pytest.mark.xfail_version(
                    pyspark=["pyspark<3.5"],
                    raises=pd._libs.tslib.OutOfBoundsDatetime,
                    reason=(
                        "versions of pandas supported by PySpark <3.5 don't allow "
                        "pd.Timestamps with out-of-bounds timestamp values"
                    ),
                ),
            ],
        ),
        param(
            "2150-01-01",
            marks=pytest.mark.notyet(["clickhouse"], raises=AssertionError),
            id="medium",
        ),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        param(lambda x: x, id="identity"),
        param(datetime.date.fromisoformat, id="fromstring"),
    ],
)
def test_date_scalar(con, value, func):
    expr = ibis.date(func(value)).name("tmp")
    result = con.execute(expr)

    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp.fromisoformat(value)


@pytest.mark.notyet(
    ["datafusion", "druid", "exasol"], raises=com.OperationNotDefinedError
)
def test_simple_unix_date_offset(con):
    s = "2023-04-07"
    d = ibis.date(s)
    expr = d.epoch_days()
    result = con.execute(expr)
    delta = datetime.date(2023, 4, 7) - datetime.date(1970, 1, 1)
    assert result == delta.days
