from __future__ import annotations

import contextlib
import datetime
import operator
import warnings
from operator import methodcaller

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.pandas.execution.temporal import day_name
from ibis.common.annotations import ValidationError

try:
    from duckdb import InvalidInputException as DuckDBInvalidInputException
except ImportError:
    DuckDBInvalidInputException = None

try:
    from polars import ComputeError as PolarsComputeError
    from polars import PanicException as PolarsPanicException

except ImportError:
    PolarsComputeError = None
    PolarsPanicException = None

try:
    from google.api_core.exceptions import BadRequest as GoogleBadRequest
except ImportError:
    GoogleBadRequest = None

try:
    from pyarrow import ArrowInvalid
except ImportError:
    ArrowInvalid = None

try:
    from clickhouse_connect.driver.exceptions import (
        InternalError as ClickhouseOperationalError,
    )
except ImportError:
    ClickhouseOperationalError = None

try:
    from impala.error import (
        HiveServer2Error as ImpalaHiveServer2Error,
    )
    from impala.error import (
        OperationalError as ImpalaOperationalError,
    )
except ImportError:
    ImpalaHiveServer2Error = ImpalaOperationalError = None


@pytest.mark.parametrize("attr", ["year", "month", "day"])
@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda c: c.date(), id="date"),
        param(
            lambda c: c.cast("date"),
            id="cast",
            marks=pytest.mark.notimpl(["impala"], raises=com.UnsupportedBackendType),
        ),
    ],
)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="Can only use .dt accessor with datetimelike values",
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
            "quarter", marks=pytest.mark.notyet(["oracle"], raises=sa.exc.DatabaseError)
        ),
        "hour",
        "minute",
        "second",
    ],
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="AttributeError: 'StringColumn' object has no attribute 'X'",
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
    ("func", "expected"),
    [
        param(
            methodcaller("year"),
            2015,
            id="year",
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
            ],
        ),
        param(
            methodcaller("month"),
            9,
            id="month",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
            ],
        ),
        param(
            methodcaller("day"),
            1,
            id="day",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
            ],
        ),
        param(
            methodcaller("hour"),
            14,
            id="hour",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=sa.exc.DatabaseError,
                    reason="ORA-30076: invalid extract field for extract source",
                ),
            ],
        ),
        param(
            methodcaller("minute"),
            48,
            id="minute",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=sa.exc.DatabaseError,
                    reason="ORA-30076: invalid extract field for extract source",
                ),
            ],
        ),
        param(
            methodcaller("second"),
            5,
            id="second",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=sa.exc.CompileError,
                    reason='No literal value renderer is available for literal value "datetime.datetime(2015, 9, 1, 14, 48, 5, 359000)" with datatype DATETIME',
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=sa.exc.DatabaseError,
                    reason="ORA-30076: invalid extract field for extract source",
                ),
            ],
        ),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
            marks=[
                pytest.mark.notimpl(
                    ["druid", "oracle"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="PySpark backend does not support extracting milliseconds.",
                ),
            ],
        ),
        param(
            lambda x: x.day_of_week.index(),
            1,
            id="day_of_week_index",
            marks=[
                pytest.mark.notimpl(
                    ["druid", "oracle"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda x: x.day_of_week.full_name(),
            "Tuesday",
            id="day_of_week_full_name",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "druid", "oracle"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
    ],
)
def test_timestamp_extract_literal(con, func, expected):
    value = ibis.timestamp("2015-09-01 14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'microsecond'",
)
@pytest.mark.notyet(
    ["pyspark"],
    raises=com.UnsupportedOperationError,
    reason="PySpark backend does not support extracting microseconds.",
)
@pytest.mark.notyet(
    ["impala"],
    raises=(ImpalaHiveServer2Error, ImpalaOperationalError),
    reason="Impala backend does not support extracting microseconds.",
)
@pytest.mark.broken(["sqlite"], raises=AssertionError)
def test_timestamp_extract_microseconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.microsecond().name("microsecond")
    result = expr.execute()
    expected = backend.default_series_rename(
        (df.timestamp_col.dt.microsecond).astype("int32")
    ).rename("microsecond")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'millisecond'",
)
@pytest.mark.notyet(
    ["pyspark"],
    raises=com.UnsupportedOperationError,
    reason="PySpark backend does not support extracting milliseconds.",
)
@pytest.mark.broken(["sqlite"], raises=AssertionError)
def test_timestamp_extract_milliseconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.millisecond().name("millisecond")
    result = expr.execute()
    expected = backend.default_series_rename(
        (df.timestamp_col.dt.microsecond // 1_000).astype("int32")
    ).rename("millisecond")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'epoch_seconds'",
)
@pytest.mark.broken(
    ["bigquery"],
    raises=GoogleBadRequest,
    reason="UNIX_SECONDS does not support DATETIME arguments",
)
@pytest.mark.xfail_version(
    pyspark=["pandas<2.1"],
    reason="test was adjusted to work with pandas 2.1 output; pyspark doesn't support pandas 2",
)
def test_timestamp_extract_epoch_seconds(backend, alltypes, df):
    expr = alltypes.timestamp_col.epoch_seconds().name("tmp")
    result = expr.execute()

    expected = backend.default_series_rename(
        df.timestamp_col.astype("datetime64[s]").astype("int64").astype("int32")
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'week_of_year'",
)
def test_timestamp_extract_week_of_year(backend, alltypes, df):
    expr = alltypes.timestamp_col.week_of_year().name("tmp")
    result = expr.execute()
    expected = backend.default_series_rename(
        df.timestamp_col.dt.isocalendar().week.astype("int32")
    )
    backend.assert_series_equal(result, expected)


PANDAS_UNITS = {
    "m": "Min",
    "ms": "L",
}


@pytest.mark.parametrize(
    "unit",
    [
        param(
            "Y",
            marks=[
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                )
            ],
        ),
        param(
            "M",
            marks=[
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                )
            ],
        ),
        param(
            "D",
            marks=[
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                )
            ],
        ),
        param(
            "W",
            marks=[
                pytest.mark.notimpl(["mysql"], raises=com.UnsupportedOperationError),
                pytest.mark.notimpl(["impala"], raises=AssertionError),
                pytest.mark.broken(["sqlite"], raises=AssertionError),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "h",
            marks=[
                pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "m",
            marks=[
                pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedOperationError),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "s",
            marks=[
                pytest.mark.notimpl(
                    ["impala", "sqlite"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "ms",
            marks=[
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "impala",
                        "mysql",
                        "pyspark",
                        "sqlite",
                        "datafusion",
                    ],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "us",
            marks=[
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "impala",
                        "mysql",
                        "pyspark",
                        "sqlite",
                        "trino",
                        "datafusion",
                    ],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.broken(
                    ["polars"],
                    raises=AssertionError,
                    reason="numpy array are different",
                ),
            ],
        ),
        param(
            "ns",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "impala",
                        "mysql",
                        "postgres",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "mssql",
                        "datafusion",
                    ],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.broken(
                    ["polars"],
                    raises=PolarsPanicException,
                    reason="attempt to calculate the remainder with a divisor of zero",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="AttributeError: 'StringColumn' object has no attribute 'truncate'",
)
def test_timestamp_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.truncate(unit).name("tmp")

    unit = PANDAS_UNITS.get(unit, unit)

    try:
        expected = df.timestamp_col.dt.floor(unit)
    except ValueError:
        expected = df.timestamp_col.dt.to_period(unit).dt.to_timestamp()

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "M",
        "D",
        param(
            "W",
            marks=[
                pytest.mark.notimpl(
                    ["mysql"],
                    raises=com.UnsupportedOperationError,
                    reason="Unsupported truncate unit W",
                ),
                pytest.mark.broken(["impala", "sqlite"], raises=AssertionError),
            ],
        ),
    ],
)
@pytest.mark.broken(
    ["polars", "druid"], reason="snaps to the UNIX epoch", raises=AssertionError
)
@pytest.mark.notimpl(
    ["datafusion", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="AttributeError: 'StringColumn' object has no attribute 'date'",
)
def test_date_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.date().truncate(unit).name("tmp")

    unit = PANDAS_UNITS.get(unit, unit)

    try:
        expected = df.timestamp_col.dt.floor(unit)
    except ValueError:
        expected = df.timestamp_col.dt.to_period(unit).dt.to_timestamp()

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
                    ["bigquery"],
                    raises=com.UnsupportedOperationError,
                    reason="BigQuery does not allow binary operation TIMESTAMP_ADD with INTERVAL offset D",
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=TypeError,
                    reason="duration() got an unexpected keyword argument 'years'",
                ),
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `add`.",
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=com.UnsupportedOperationError,
                    reason="year not implemented",
                ),
            ],
        ),
        param("Q", pd.offsets.DateOffset, marks=pytest.mark.xfail),
        param(
            "M",
            pd.offsets.DateOffset,
            # TODO - DateOffset - #2553
            marks=[
                pytest.mark.notimpl(
                    ["bigquery"],
                    raises=com.UnsupportedOperationError,
                    reason="BigQuery does not allow binary operation TIMESTAMP_ADD with INTERVAL offset M",
                ),
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `add`.",
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=TypeError,
                    reason="duration() got an unexpected keyword argument 'months'",
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=com.UnsupportedOperationError,
                    reason="month not implemented",
                ),
            ],
        ),
        param(
            "W",
            pd.offsets.DateOffset,
            # TODO - DateOffset - #2553
            marks=[
                pytest.mark.notimpl(
                    ["bigquery"],
                    raises=com.UnsupportedOperationError,
                    reason="BigQuery does not allow extracting date part `IntervalUnit.WEEK` from intervals",
                ),
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `add`.",
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=com.UnsupportedOperationError,
                    reason="week not implemented",
                ),
            ],
        ),
        param(
            "D",
            pd.offsets.DateOffset,
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                ),
            ],
        ),
        param(
            "h",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                )
            ],
        ),
        param(
            "m",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                )
            ],
        ),
        param(
            "s",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="Interval from integer column is unsupported for the PySpark backend.",
                ),
            ],
        ),
        param(
            "ms",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "clickhouse"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedArgumentError,
                    reason="Interval unit \"ms\" is not allowed. Allowed units are: ['Y', 'W', 'M', 'D', 'h', 'm', 's']",
                ),
            ],
        ),
        param(
            "us",
            pd.Timedelta,
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse"], raises=com.UnsupportedOperationError
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedArgumentError,
                    reason="Interval unit \"us\" is not allowed. Allowed units are: ['Y', 'W', 'M', 'D', 'h', 'm', 's']",
                ),
                pytest.mark.notimpl(
                    ["trino"],
                    raises=AssertionError,
                    reason="we're dropping microseconds to ensure results consistent with pandas",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["datafusion", "sqlite", "snowflake", "mssql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=ValidationError,
    reason="Given argument with datatype interval('h') is not implicitly castable to string",
)
def test_integer_to_interval_timestamp(
    backend, con, alltypes, df, unit, displacement_type
):
    interval = alltypes.int_col.to_interval(unit=unit)
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
    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize(
    "unit",
    [
        param(
            "Y",
            marks=pytest.mark.notyet(["trino"], raises=com.UnsupportedOperationError),
        ),
        param("Q", marks=pytest.mark.xfail),
        param(
            "M",
            marks=pytest.mark.notyet(["trino"], raises=com.UnsupportedOperationError),
        ),
        param(
            "W",
            marks=pytest.mark.notyet(["trino"], raises=com.UnsupportedOperationError),
        ),
        "D",
    ],
)
# TODO - DateOffset - #2553
@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "impala",
        "mysql",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "druid",
        "oracle",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    [
        "pyspark",
    ],
    raises=com.UnsupportedOperationError,
    reason="Interval from integer column is unsupported for the PySpark backend.",
)
@pytest.mark.notimpl(
    [
        "sqlite",
    ],
    raises=(com.UnsupportedOperationError, com.OperationNotDefinedError),
    reason="Handling unsupported op error for DateAdd with weeks",
)
def test_integer_to_interval_date(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    array = alltypes.date_string_col.split("/")
    month, day, year = array[0], array[1], array[2]
    date_col = expr = ibis.literal("-").join(["20" + year, month, day]).cast("date")
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
        expected = pd.to_datetime(df.date_string_col) + offset

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected.map(lambda ts: ts.normalize()))


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
                pytest.mark.notimpl(
                    ["sqlite"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=ValidationError,
                    reason="Given argument with datatype interval('D') is not implicitly castable to string",
                ),
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
                    [
                        "clickhouse",
                        "dask",
                        "impala",
                        "mysql",
                        "pandas",
                        "postgres",
                        "snowflake",
                        "sqlite",
                        "bigquery",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=ValidationError,
                    reason="Given argument with datatype interval('D') is not implicitly castable to string",
                ),
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
                    [
                        "clickhouse",
                        "sqlite",
                        "postgres",
                        "polars",
                        "mysql",
                        "impala",
                        "snowflake",
                        "bigquery",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=ValidationError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            lambda t, _: t.timestamp_col - ibis.interval(days=17),
            lambda t, _: t.timestamp_col - pd.Timedelta(days=17),
            id="timestamp-subtract-interval",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'IntervalScalar'",
                ),
                pytest.mark.notimpl(
                    ["sqlite"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, _: t.timestamp_col.date() + ibis.interval(days=4),
            lambda t, _: t.timestamp_col.dt.floor("d") + pd.Timedelta(days=4),
            id="date-add-interval",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'StringColumn' object has no attribute 'date'",
                ),
            ],
        ),
        param(
            lambda t, _: t.timestamp_col.date() - ibis.interval(days=14),
            lambda t, _: t.timestamp_col.dt.floor("d") - pd.Timedelta(days=14),
            id="date-subtract-interval",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'StringColumn' object has no attribute 'date'",
                ),
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
                    ["bigquery", "snowflake", "sqlite"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedOperationError,
                    reason="PySpark backend does not support TimestampDiff as there is no timedelta type.",
                ),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=ValidationError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'TimestampScalar'",
                ),
                pytest.mark.xfail_version(
                    duckdb=["duckdb>=0.8.0"],
                    raises=AssertionError,
                    reason="duckdb 0.8.0 returns DateOffset columns",
                ),
                pytest.mark.broken(
                    ["trino"],
                    raises=AssertionError,
                    reason="doesn't match pandas results, unclear what the issue is, perhaps timezones",
                ),
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
                pytest.mark.xfail_version(
                    pyspark=["pyspark<3.3"],
                    raises=AttributeError,
                    reason="DayTimeIntervalType added in pyspark 3.3",
                ),
                pytest.mark.notimpl(["bigquery"], raises=com.OperationNotDefinedError),
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'StringColumn' object has no attribute 'date'",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["mssql", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(["datafusion"], raises=BaseException)
def test_temporal_binop(backend, con, alltypes, df, expr_fn, expected_fn):
    expr = expr_fn(alltypes, backend).name("tmp")
    expected = expected_fn(df, backend)

    result = con.execute(expr)
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected.astype(result.dtype))


plus = lambda t, td: t.timestamp_col + pd.Timedelta(td)
minus = lambda t, td: t.timestamp_col - pd.Timedelta(td)


@pytest.mark.parametrize(
    ("timedelta", "temporal_fn"),
    [
        param(
            "36500d",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
                pytest.mark.broken(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
                pytest.mark.broken(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
            ],
        ),
        param(
            "5W",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            "3d",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            "2h",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            "3m",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            "10s",
            plus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=AssertionError,
                    reason="alltypes.timestamp_col is represented as string",
                ),
            ],
        ),
        param(
            "36500d",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
                pytest.mark.broken(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
                pytest.mark.broken(
                    ["clickhouse"],
                    raises=AssertionError,
                    reason="DateTime column overflows, should use DateTime64",
                ),
            ],
        ),
        param(
            "5W",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
            ],
        ),
        param(
            "3d",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
            ],
        ),
        param(
            "2h",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
            ],
        ),
        param(
            "3m",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
            ],
        ),
        param(
            "10s",
            minus,
            marks=[
                pytest.mark.broken(
                    ["druid"],
                    raises=TypeError,
                    reason="unsupported operand type(s) for -: 'StringColumn' and 'Timedelta'",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["datafusion", "sqlite", "mssql", "oracle"], raises=com.OperationNotDefinedError
)
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
    ["polars"],
    raises=TypeError,
    reason="casting a timezone aware value to timezone aware dtype was removed",
)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="Can only use .dt accessor with datetimelike values",
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
    [
        param(
            "gt",
            marks=[
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `gt`.",
                ),
                pytest.mark.notimpl(
                    ["pandas"],
                    raises=TypeError,
                    reason="Invalid comparison between dtype=datetime64[ns, UTC] and datetime",
                ),
            ],
        ),
        param(
            "ge",
            marks=[
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `ge`.",
                ),
                pytest.mark.notimpl(
                    ["pandas"],
                    raises=TypeError,
                    reason="Invalid comparison between dtype=datetime64[ns, UTC] and datetime",
                ),
            ],
        ),
        param(
            "lt",
            marks=[
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `lt`.",
                ),
                pytest.mark.notimpl(
                    ["pandas"],
                    raises=TypeError,
                    reason="Invalid comparison between dtype=datetime64[ns, UTC] and datetime",
                ),
            ],
        ),
        param(
            "le",
            marks=[
                pytest.mark.notimpl(
                    ["dask"],
                    raises=ValueError,
                    reason="Metadata inference failed in `le`.",
                ),
                pytest.mark.notimpl(
                    ["pandas"],
                    raises=TypeError,
                    reason="Invalid comparison between dtype=datetime64[ns, UTC] and datetime",
                ),
            ],
        ),
        "eq",
        "ne",
    ],
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="Can only use .dt accessor with datetimelike values",
)
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


@pytest.mark.notimpl(
    ["sqlite", "snowflake", "mssql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'date'",
)
def test_interval_add_cast_scalar(backend, alltypes):
    timestamp_date = alltypes.timestamp_col.date()
    delta = ibis.literal(10).cast("interval('D')")
    expr = (timestamp_date + delta).name("result")
    result = expr.execute()
    expected = timestamp_date.name("result").execute() + pd.Timedelta(10, unit="D")
    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.never(
    ["pyspark"], reason="PySpark does not support casting columns to intervals"
)
@pytest.mark.notimpl(
    ["sqlite", "snowflake", "mssql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'date'",
)
def test_interval_add_cast_column(backend, alltypes, df):
    timestamp_date = alltypes.timestamp_col.date()
    delta = alltypes.bigint_col.cast("interval('D')")
    expr = alltypes["id", (timestamp_date + delta).name("tmp")]
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
                t.mutate(suffix="%d")
                .select(
                    [
                        lambda t: t.timestamp_col.strftime("%Y%m" + t.suffix).name(
                            "formatted"
                        )
                    ]
                )
                .formatted
            ),
            "%Y%m%d",
            marks=[
                pytest.mark.notimpl(
                    [
                        "pandas",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    [
                        "pyspark",
                    ],
                    raises=AttributeError,
                    reason="'StringConcat' object has no attribute 'value'",
                ),
                pytest.mark.notimpl(
                    [
                        "postgres",
                        "snowflake",
                    ],
                    raises=AttributeError,
                    reason="Neither 'concat' object nor 'Comparator' object has an attribute 'value'",
                ),
                pytest.mark.notimpl(
                    [
                        "polars",
                    ],
                    raises=com.UnsupportedArgumentError,
                    reason="Polars does not support columnar argument StringConcat()",
                ),
                pytest.mark.notyet(["dask"], raises=com.OperationNotDefinedError),
                pytest.mark.broken(
                    ["impala"],
                    raises=AttributeError,
                    reason="'StringConcat' object has no attribute 'value'",
                ),
                pytest.mark.notyet(
                    ["duckdb"],
                    raises=com.UnsupportedOperationError,
                    reason=(
                        "DuckDB format_str must be a literal `str`; got "
                        "<class 'ibis.expr.operations.strings.StringConcat'>"
                    ),
                ),
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
    [
        "datafusion",
        "mssql",
        "oracle",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'strftime'",
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
                    ["pyspark"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit ms. Supported unit is s.",
                ),
                pytest.mark.notimpl(
                    ["clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`ms` unit is not supported!",
                ),
            ],
        ),
        param(
            "us",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit us. Supported unit is s.",
                ),
                pytest.mark.notimpl(
                    ["duckdb", "mssql", "clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`us` unit is not supported!",
                ),
            ],
        ),
        param(
            "ns",
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.UnsupportedArgumentError,
                    reason="PySpark backend does not support timestamp from unix time with unit ms. Supported unit is s.",
                ),
                pytest.mark.notimpl(
                    ["duckdb", "mssql", "clickhouse"],
                    raises=com.UnsupportedOperationError,
                    reason="`ms` unit is not supported!",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(
    ["datafusion", "mysql", "postgres", "sqlite", "druid", "oracle"],
    raises=com.OperationNotDefinedError,
)
def test_integer_to_timestamp(backend, con, unit):
    backend_unit = backend.returned_timestamp_unit
    factor = unit_factors[unit]

    pandas_ts = pd.Timestamp("2018-04-13 09:54:11.872832").floor(unit).value

    # convert the timestamp to the input unit being tested
    int_expr = ibis.literal(pandas_ts // factor)
    expr = int_expr.to_timestamp(unit).name("tmp")
    result = con.execute(expr)
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
                    ["pyspark"],
                    reason=(
                        "datetime formatting style not supported. "
                        "Test failed with message: NaTType does not support strftime"
                    ),
                    raises=ValueError,
                ),
                pytest.mark.never(
                    ["snowflake"],
                    reason=(
                        "(snowflake.connector.errors.ProgrammingError) 100096 (22007): "
                        "Can't parse '11/01/10' as timestamp with format '%m/%d/%y'"
                    ),
                    raises=sa.exc.ProgrammingError,
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
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.never(
                    ["polars"],
                    reason="datetime formatting style not supported",
                    raises=PolarsComputeError,
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
    [
        "dask",
        "pandas",
        "postgres",
        "clickhouse",
        "sqlite",
        "impala",
        "datafusion",
        "mssql",
        "druid",
        "oracle",
    ],
    raises=com.OperationNotDefinedError,
)
def test_string_to_timestamp(alltypes, fmt):
    table = alltypes
    result = table.mutate(date=table.date_string_col.to_timestamp(fmt)).execute()

    # TEST: do we get the same date out, that we put in?
    # format string assumes that we are using pandas' strftime
    for i, val in enumerate(result["date"]):
        assert val.strftime("%m/%d/%y") == result["date_string_col"][i]


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
@pytest.mark.notimpl(["mssql", "druid", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["impala"], raises=com.UnsupportedBackendType)
def test_day_of_week_scalar(con, date, expected_index, expected_day):
    expr = ibis.literal(date).cast(dt.date)
    result_index = con.execute(expr.day_of_week.index().name("tmp"))
    assert result_index == expected_index

    result_day = con.execute(expr.day_of_week.full_name().name("tmp"))
    assert result_day.lower() == expected_day.lower()


@pytest.mark.notimpl(["mssql", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="StringColumn' object has no attribute 'day_of_week'",
)
def test_day_of_week_column(backend, alltypes, df):
    expr = alltypes.timestamp_col.day_of_week

    result_index = expr.index().name("tmp").execute()
    expected_index = df.timestamp_col.dt.dayofweek.astype("int16")

    backend.assert_series_equal(result_index, expected_index, check_names=False)

    result_day = expr.full_name().name("tmp").execute()
    expected_day = day_name(df.timestamp_col.dt)

    backend.assert_series_equal(result_day, expected_day, check_names=False)


@pytest.mark.parametrize(
    ("day_of_week_expr", "day_of_week_pandas"),
    [
        param(
            lambda t: t.timestamp_col.day_of_week.index().count(),
            lambda s: s.dt.dayofweek.count(),
            id="day_of_week_index",
        ),
        param(
            lambda t: t.timestamp_col.day_of_week.full_name().length().sum(),
            lambda s: day_name(s.dt).str.len().sum(),
            id="day_of_week_full_name",
            marks=[pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError)],
        ),
    ],
)
@pytest.mark.notimpl(["oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'day_of_week'",
)
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

    # FIXME(#1536): Pandas backend should use query.schema().apply_to
    backend.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.notimpl(
    ["datafusion", "druid", "oracle"], raises=com.OperationNotDefinedError
)
def test_now(con):
    expr = ibis.now()
    result = con.execute(expr.name("tmp"))
    assert isinstance(result, datetime.datetime)


@pytest.mark.notimpl(["polars"], reason="assert 1 == 5", raises=AssertionError)
@pytest.mark.notimpl(
    ["datafusion", "druid", "oracle"], raises=com.OperationNotDefinedError
)
def test_now_from_projection(alltypes):
    n = 2
    expr = alltypes.select(now=ibis.now()).limit(n)
    result = expr.execute()
    ts = result.now
    assert len(result) == n
    assert ts.nunique() == 1
    assert ~pd.isna(ts.iat[0])


DATE_BACKEND_TYPES = {
    "bigquery": "DATE",
    "clickhouse": "Date",
    "snowflake": "DATE",
    "sqlite": "text",
    "trino": "date",
    "duckdb": "DATE",
    "postgres": "date",
}


@pytest.mark.notimpl(
    ["pandas", "datafusion", "dask", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["druid"], raises=sa.exc.ProgrammingError, reason="SQL parse failed"
)
@pytest.mark.notimpl(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-00936 missing expression"
)
@pytest.mark.broken(
    ["mysql"],
    raises=sa.exc.ProgrammingError,
    reason=(
        '(pymysql.err.ProgrammingError) (1064, "You have an error in your SQL syntax; '
        "check the manual that corresponds to your MariaDB server version for "
        "the right syntax to use near ' 2, 4) AS `DateFromYMD(2022, 2, 4)`' at line 1\")"
        "[SQL: SELECT date(%(param_1)s, %(param_2)s, %(param_3)s) AS `DateFromYMD(2022, 2, 4)`]"
    ),
)
@pytest.mark.notyet(["impala"], raises=com.OperationNotDefinedError)
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
    "duckdb": "TIMESTAMP",
    "postgres": "timestamp without time zone",
}


@pytest.mark.notimpl(
    ["pandas", "datafusion", "dask", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason=(
        "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
        "(org.apache.calcite.tools.ValidationException): org.apache.calcite.runtime.CalciteContextException: "
        "From line 1, column 8 to line 1, column 44: No match found for function signature "
        "make_timestamp(<NUMERIC>, <NUMERIC>, <NUMERIC>, <NUMERIC>, <NUMERIC>, <NUMERIC>)"
    ),
)
@pytest.mark.broken(
    ["mysql"],
    raises=sa.exc.OperationalError,
    reason="(pymysql.err.OperationalError) (1305, 'FUNCTION ibis_testing.make_timestamp does not exist')",
)
@pytest.mark.notimpl(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-00904: MAKE TIMESTAMP invalid"
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
    ["pandas", "datafusion", "mysql", "dask", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["mysql"],
    raises=sa.exc.OperationalError,
    reason="FUNCTION ibis_testing.make_timestamp does not exist",
)
@pytest.mark.notimpl(
    ["sqlite"],
    raises=com.UnsupportedOperationError,
    reason=(
        "Unable to cast from Timestamp(timezone=None, scale=None, nullable=True) to "
        "Timestamp(timezone='***', scale=None, nullable=True)."
    ),
)
@pytest.mark.notyet(["impala"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-00904: MAKE TIMESTAMP invalid"
)
@pytest.mark.parametrize(
    ("timezone", "expected"),
    [
        param(
            "Europe/London",
            "2022-02-04 16:20:00GMT",
            id="name",
            marks=[pytest.mark.broken(["mssql"], raises=TypeError)],
        ),
        param(
            "PST8PDT",
            "2022-02-04 08:20:00PST",
            # The time zone for Berkeley, California.
            id="iso",
            marks=[pytest.mark.broken(["mssql"], raises=TypeError)],
        ),
    ],
)
@pytest.mark.notimpl(
    ["bigquery"],
    "BigQuery does not support timestamps with timezones other than 'UTC'",
    raises=TypeError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason=(
        "No match found for function signature make_timestamp(<NUMERIC>, <NUMERIC>, "
        "<NUMERIC>, <NUMERIC>, <NUMERIC>, <NUMERIC>)"
    ),
)
def test_timestamp_with_timezone_literal(con, backend, timezone, expected):
    expr = ibis.timestamp(2022, 2, 4, 16, 20, 0).cast(dt.Timestamp(timezone=timezone))
    result = con.execute(expr)
    if not isinstance(result, str):
        result = result.strftime("%Y-%m-%d %H:%M:%S%Z")
    assert result == expected


TIME_BACKEND_TYPES = {
    "bigquery": "TIME",
    "snowflake": "TIME",
    "sqlite": "text",
    "trino": "time(3)",
    "duckdb": "TIME",
    "postgres": "time without time zone",
}


@pytest.mark.notimpl(
    [
        "pandas",
        "datafusion",
        "dask",
        "pyspark",
        "polars",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(["clickhouse", "impala"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
@pytest.mark.broken(
    [
        "mysql",
    ],
    raises=sa.exc.ProgrammingError,
    reason=(
        '(pymysql.err.ProgrammingError) (1064, "You have an error in your SQL syntax; check the manual that '
        "corresponds to your MariaDB server version for the right syntax to use near ' 20, 0) AS "
        "`TimeFromHMS(16, 20, 0)`' at line 1\")"
        "[SQL: SELECT time(%(param_1)s, %(param_2)s, %(param_3)s) AS `TimeFromHMS(16, 20, 0)`]"
    ),
)
@pytest.mark.broken(
    ["druid"], raises=sa.exc.ProgrammingError, reason="SQL parse failed"
)
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
    ["clickhouse", "impala", "pyspark"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't have a time datatype",
)
@pytest.mark.notyet(
    ["druid"],
    raises=sa.exc.CompileError,
    reason="druid sqlalchemy dialect fails to compile datetime types",
)
@pytest.mark.broken(
    ["sqlite"], raises=AssertionError, reason="SQLite returns Timedelta from execution"
)
@pytest.mark.notimpl(
    ["dask", "datafusion", "pandas"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(["oracle"], raises=sa.exc.DatabaseError)
@pytest.mark.parametrize(
    "microsecond",
    [
        0,
        param(
            561021,
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "mysql"],
                    raises=AssertionError,
                    reason="doesn't have enough precision to capture microseconds",
                ),
                pytest.mark.notyet(
                    ["trino"],
                    raises=AssertionError,
                    reason="has enough precision, but sqlalchemy dialect drops them",
                ),
            ],
        ),
    ],
    ids=["second", "subsecond"],
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
    "duckdb": "INTERVAL",
    "postgres": "interval",
}


@pytest.mark.broken(
    ["snowflake"],
    "(snowflake.connector.errors.ProgrammingError) 001007 (22023): SQL compilation error:"
    "invalid type [CAST(INTERVAL_LITERAL('second', '1') AS VARIANT)] for parameter 'TO_VARIANT'",
    raises=sa.exc.ProgrammingError,
)
@pytest.mark.broken(
    ["druid"],
    'No literal value renderer is available for literal value "1" with datatype DATETIME',
    raises=sa.exc.CompileError,
)
@pytest.mark.broken(
    ["impala"],
    "AnalysisException: Syntax error in line 1: SELECT typeof(INTERVAL 1 SECOND) AS `TypeOf(1)` "
    "Encountered: ) Expected: +",
    raises=ImpalaHiveServer2Error,
)
@pytest.mark.broken(
    ["pyspark"],
    "Invalid argument, not a string or column: 1000000000 of type <class 'int'>. For column literals, "
    "use 'lit', 'array', 'struct' or 'create_map' function.",
    raises=TypeError,
)
@pytest.mark.broken(
    ["mysql"],
    "The backend implementation is broken. "
    "If SQLAlchemy < 2 is installed, test fails with the following exception:"
    "AttributeError: 'TextClause' object has no attribute 'label'"
    "If SQLAlchemy >=2 is installed, test fails with the following exception:"
    "NotImplementedError",
    raises=(NotImplementedError, AttributeError),
)
@pytest.mark.broken(
    ["bigquery"], reason="BigQuery returns DateOffset arrays", raises=AssertionError
)
@pytest.mark.xfail_version(
    datafusion=["datafusion"],
    raises=Exception,
    reason='This feature is not implemented: Can\'t create a scalar from array of type "Duration(Second)"',
)
@pytest.mark.notyet(
    ["clickhouse"],
    reason="Driver doesn't know how to handle intervals",
    raises=ClickhouseOperationalError,
)
@pytest.mark.xfail_version(
    duckdb=["duckdb>=0.8.0"],
    raises=AssertionError,
    reason="duckdb 0.8.0 returns DateOffset columns",
)
def test_interval_literal(con, backend):
    expr = ibis.interval(1, unit="s")
    result = con.execute(expr)
    assert str(result) == "0 days 00:00:01"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == INTERVAL_BACKEND_TYPES[backend_name]


@pytest.mark.notimpl(
    ["pandas", "datafusion", "dask", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["mysql"],
    raises=sa.exc.ProgrammingError,
    reason=(
        '(pymysql.err.ProgrammingError) (1064, "You have an error in your SQL syntax; check the manual '
        "that corresponds to your MariaDB server version for the right syntax to use near "
        "' CAST(EXTRACT(month FROM t0.timestamp_col) AS SIGNED INTEGER), CAST(EXTRACT(d...' at line 1\")"
    ),
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="'StringColumn' object has no attribute 'year'",
)
@pytest.mark.broken(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-00936: missing expression"
)
@pytest.mark.notyet(["impala"], raises=com.OperationNotDefinedError)
def test_date_column_from_ymd(con, alltypes, df):
    c = alltypes.timestamp_col
    expr = ibis.date(c.year(), c.month(), c.day())
    tbl = alltypes[expr.name("timestamp_col")]
    result = con.execute(tbl)

    golden = df.timestamp_col.dt.date.astype(result.timestamp_col.dtype)
    tm.assert_series_equal(golden, result.timestamp_col)


@pytest.mark.notimpl(
    ["pandas", "datafusion", "dask", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="StringColumn' object has no attribute 'year'",
)
@pytest.mark.broken(
    ["mysql"],
    raises=sa.exc.OperationalError,
    reason="(pymysql.err.OperationalError) (1305, 'FUNCTION ibis_testing.make_timestamp does not exist')",
)
@pytest.mark.notimpl(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-00904 make timestamp invalid"
)
@pytest.mark.notyet(["impala"], raises=com.OperationNotDefinedError)
def test_timestamp_column_from_ymdhms(con, alltypes, df):
    c = alltypes.timestamp_col
    expr = ibis.timestamp(
        c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second()
    )
    tbl = alltypes[expr.name("timestamp_col")]
    result = con.execute(tbl)

    golden = df.timestamp_col.dt.floor("s").astype(result.timestamp_col.dtype)
    tm.assert_series_equal(golden, result.timestamp_col)


@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason=(
        "(pydruid.db.exceptions.ProgrammingError) Plan validation failed "
        "(org.apache.calcite.tools.ValidationException): "
        "java.lang.UnsupportedOperationException: class org.apache.calcite.sql.SqlIdentifier: LONG"
    ),
)
@pytest.mark.notimpl(["impala"], raises=com.UnsupportedBackendType)
@pytest.mark.notimpl(
    ["oracle"], raises=sa.exc.DatabaseError, reason="ORA-01861 literal does not match"
)
def test_date_scalar_from_iso(con):
    expr = ibis.literal("2022-02-24")
    expr2 = ibis.date(expr)

    result = con.execute(expr2)
    assert result.strftime("%Y-%m-%d") == "2022-02-24"


@pytest.mark.notimpl(["mssql", "druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason="java.lang.UnsupportedOperationException: class org.apache.calcite.sql.SqlIdentifier: STRING",
)
@pytest.mark.notimpl(["impala"], raises=com.UnsupportedBackendType)
@pytest.mark.notyet(
    ["oracle"],
    raises=sa.exc.DatabaseError,
    reason="ORA-22849 type CLOB is not supported",
)
def test_date_column_from_iso(con, alltypes, df):
    expr = (
        alltypes.year.cast("string")
        + "-"
        + alltypes.month.cast("string").lpad(2, "0")
        + "-13"
    )
    expr = ibis.date(expr)

    result = con.execute(expr.name("tmp"))
    golden = df.year.astype(str) + "-" + df.month.astype(str).str.rjust(2, "0") + "-13"
    actual = result.dt.strftime("%Y-%m-%d")
    tm.assert_series_equal(golden.rename("tmp"), actual.rename("tmp"))


@pytest.mark.notimpl(["druid", "oracle"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["pyspark"],
    raises=com.UnsupportedOperationError,
    reason=" PySpark backend does not support extracting milliseconds.",
)
@pytest.mark.notyet(
    ["pyspark"],
    raises=com.UnsupportedOperationError,
    reason="PySpark backend does not support extracting milliseconds.",
)
def test_timestamp_extract_milliseconds_with_big_value(con):
    timestamp = ibis.timestamp("2021-01-01 01:30:59.333456")
    millis = timestamp.millisecond()
    result = con.execute(millis.name("tmp"))
    assert result == 333


@pytest.mark.notimpl(
    ["datafusion"],
    raises=Exception,
    reason=(
        "This feature is not implemented: Unsupported CAST from Int32 to Timestamp(Nanosecond, None)"
    ),
)
@pytest.mark.notimpl(
    ["oracle"],
    raises=sa.exc.DatabaseError,
    reason="ORA-00932",
)
@pytest.mark.broken(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason="No match found for function signature to_timestamp(<NUMERIC>)",
)
def test_integer_cast_to_timestamp_column(backend, alltypes, df):
    expr = alltypes.int_col.cast("timestamp")
    expected = pd.to_datetime(df.int_col, unit="s").rename(expr.get_name())
    result = expr.execute()
    backend.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.notimpl(
    ["datafusion"],
    raises=Exception,
    reason=(
        "Internal error: Invalid aggregate expression 'CAST(MIN(functional_alltypes.int_col) "
        "AS Timestamp(Nanosecond, None)) AS tmp'. This was likely caused by a bug in "
        "DataFusion's code and we would welcome that you file an bug report in our issue tracker"
    ),
)
@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason="No match found for function signature to_timestamp(<NUMERIC>)",
)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
def test_integer_cast_to_timestamp_scalar(alltypes, df):
    expr = alltypes.int_col.min().cast("timestamp")
    result = expr.execute()
    expected = pd.to_datetime(df.int_col.min(), unit="s")
    assert result == expected


@pytest.mark.broken(
    ["clickhouse"],
    raises=AssertionError,
)
@pytest.mark.notimpl(
    ["polars", "druid"],
    reason="Arrow backends assume a ns resolution timestamps",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["druid"],
    reason='No literal value renderer is available for literal value "datetime.datetime(2419, 10, 11, 10, 10, 25)" with datatype DATETIME',
    raises=sa.exc.CompileError,
)
@pytest.mark.notimpl(
    ["polars"],
    raises=(AssertionError, PolarsComputeError),
    reason=(
        "Casting from timestamp[us] to timestamp[ns] would "
        "result in out of bounds timestamp: 14193569425000000"
    ),
)
@pytest.mark.notyet(
    ["pyspark"],
    reason="PySpark doesn't handle big timestamps",
    raises=pd.errors.OutOfBoundsDatetime,
)
def test_big_timestamp(con):
    # TODO: test with a timezone
    value = ibis.timestamp("2419-10-11 10:10:25")
    result = con.execute(value.name("tmp"))
    expected = datetime.datetime(2419, 10, 11, 10, 10, 25)
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


@pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["druid"],
    raises=sa.exc.CompileError,
    reason='No literal value renderer is available for literal value "datetime.date(2010, 11, 1)" with datatype DATE',
)
@pytest.mark.notyet(
    ["impala"],
    reason="impala doesn't support dates",
    raises=com.UnsupportedBackendType,
)
@pytest.mark.notimpl(["oracle"], raises=sa.exc.DatabaseError)
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


@pytest.mark.broken(
    ["clickhouse"],
    reason="returns incorrect results",
    raises=AssertionError,
)
@pytest.mark.notimpl(
    ["druid"],
    raises=sa.exc.CompileError,
    reason='No literal value renderer is available for literal value "datetime.datetime(4567, 1, 1, 0, 0)" with datatype DATETIME',
)
@pytest.mark.notimpl(["pyspark"], raises=pd.errors.OutOfBoundsDatetime)
@pytest.mark.notimpl(
    ["polars"],
    raises=PolarsPanicException,
    reason=(
        "called `Result::unwrap()` on an `Err` value: PyErr { type: <class 'OverflowError'>, "
        "value: OverflowError('int too big to convert'), traceback: None }"
    ),
)
def test_large_timestamp(con):
    huge_timestamp = datetime.datetime(year=4567, month=1, day=1)
    expr = ibis.timestamp("4567-01-01 00:00:00")
    result = con.execute(expr)
    assert result.replace(tzinfo=None) == huge_timestamp


@pytest.mark.parametrize(
    ("ts", "scale", "unit"),
    [
        param(
            "2023-01-07 13:20:05.561",
            3,
            "ms",
            id="ms",
            marks=pytest.mark.broken(
                ["mssql"], reason="incorrect result", raises=AssertionError
            ),
        ),
        param(
            "2023-01-07 13:20:05.561021",
            6,
            "us",
            id="us",
            marks=[
                pytest.mark.broken(
                    ["mssql"], reason="incorrect result", raises=AssertionError
                ),
                pytest.mark.notyet(
                    ["sqlite"],
                    reason="doesn't support microseconds",
                    raises=AssertionError,
                ),
            ],
        ),
        param(
            "2023-01-07 13:20:05.561000231",
            9,
            "ns",
            id="ns",
            marks=[
                pytest.mark.broken(
                    ["duckdb", "impala", "pyspark", "trino"],
                    reason="drivers appear to truncate nanos",
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
                    raises=sa.exc.OperationalError,
                ),
                pytest.mark.notyet(
                    ["bigquery"],
                    reason=(
                        "doesn't support nanoseconds. "
                        "Server returns: 400 Invalid timestamp: '2023-01-07 13:20:05.561000231'"
                    ),
                    raises=GoogleBadRequest,
                ),
            ],
        ),
    ],
)
@pytest.mark.notyet(["mysql"], raises=AssertionError)
@pytest.mark.broken(
    ["druid"],
    raises=sa.exc.ProgrammingError,
    reason=(
        "java.lang.UnsupportedOperationException: class "
        "org.apache.calcite.sql.SqlIdentifier: LONG"
    ),
)
@pytest.mark.notimpl(
    ["oracle"],
    raises=sa.exc.DatabaseError,
    reason="ORA-01843: invalid month was specified",
)
def test_timestamp_precision_output(con, ts, scale, unit):
    dtype = dt.Timestamp(scale=scale)
    expr = ibis.literal(ts).cast(dtype)
    result = con.execute(expr)
    expected = pd.Timestamp(ts).floor(unit)
    assert result == expected


@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "druid",
        "flink",
        "impala",
        "oracle",
        "pandas",
        "polars",
        "pyspark",
        "sqlite",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["postgres"],
    reason="postgres doesn't have any easy way to accurately compute the delta in specific units",
    raises=com.OperationNotDefinedError,
)
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
                    raises=NotImplementedError,
                    reason="time types not yet implemented in ibis for the clickhouse backend",
                )
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
                    ["mysql"],
                    raises=com.OperationNotDefinedError,
                    reason="timestampdiff rounds after subtraction and mysql doesn't have a date_trunc function",
                )
            ],
        ),
    ],
)
def test_delta(con, start, end, unit, expected):
    expr = end.delta(start, unit)
    assert con.execute(expr) == expected


@pytest.mark.notimpl(
    [
        "bigquery",
        "dask",
        "flink",
        "impala",
        "mysql",
        "oracle",
        "pandas",
        "pyspark",
        "sqlite",
        "trino",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="Druid tests load timestamp_col as a string currently",
)
@pytest.mark.parametrize(
    "kws, pd_freq",
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
                    raises=sa.exc.ProgrammingError,
                    reason="snowflake doesn't support sub-second interval precision",
                ),
                pytest.mark.notimpl(
                    ["datafusion"],
                    raises=com.UnsupportedOperationError,
                    reason="backend doesn't support sub-second interval precision",
                ),
            ],
            id="milliseconds",
        ),
        param(
            {"seconds": 2},
            "2s",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion"],
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
                    ["datafusion"],
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
                    ["datafusion"],
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
                    ["datafusion"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="days",
        ),
    ],
)
def test_timestamp_bucket(backend, kws, pd_freq):
    ts = backend.functional_alltypes.timestamp_col.name("ts").execute()
    res = backend.functional_alltypes.timestamp_col.bucket(**kws).name("ts").execute()
    sol = ts.dt.floor(pd_freq)
    backend.assert_series_equal(res, sol)


@pytest.mark.notimpl(
    [
        "bigquery",
        "dask",
        "datafusion",
        "flink",
        "impala",
        "mysql",
        "oracle",
        "pandas",
        "pyspark",
        "sqlite",
        "trino",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["druid"],
    raises=AttributeError,
    reason="Druid tests load timestamp_col as a string currently",
)
@pytest.mark.notimpl(
    ["clickhouse", "mssql", "snowflake"],
    reason="offset arg not supported",
    raises=com.UnsupportedOperationError,
)
@pytest.mark.parametrize("offset_mins", [2, -2], ids=["pos", "neg"])
def test_timestamp_bucket_offset(backend, offset_mins):
    ts = backend.functional_alltypes.timestamp_col.name("ts")
    expr = ts.bucket(minutes=5, offset=ibis.interval(minutes=offset_mins)).name("ts")
    res = expr.execute().astype("datetime64[ns]")
    td = pd.Timedelta(minutes=offset_mins)
    sol = ((ts.execute() - td).dt.floor("300s") + td).astype("datetime64[ns]")
    backend.assert_series_equal(res, sol)
