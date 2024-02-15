from __future__ import annotations

import contextlib
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    OracleDatabaseError,
    PsycoPg2InternalError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
)
from ibis.common.annotations import ValidationError


@pytest.mark.parametrize(
    ("text_value", "expected_types"),
    [
        param(
            "STRING",
            {
                "bigquery": "STRING",
                "clickhouse": "String",
                "snowflake": "VARCHAR",
                "sqlite": "text",
                "trino": "varchar(6)",
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
                "risingwave": "text",
                "flink": "CHAR(6) NOT NULL",
            },
            id="string",
        ),
        param(
            "STRI'NG",
            {
                "bigquery": "STRING",
                "clickhouse": "String",
                "snowflake": "VARCHAR",
                "sqlite": "text",
                "trino": "varchar(7)",
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
                "risingwave": "text",
                "flink": "CHAR(7) NOT NULL",
            },
            id="string-quote1",
            marks=[
                pytest.mark.broken(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-01741: illegal zero length identifier",
                ),
                pytest.mark.broken(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason='sql parser error: Expected end of statement, found: "NG\'" at line:1, column:31 Near "SELECT \'STRI"NG\' AS "\'STRI""',
                ),
            ],
        ),
        param(
            'STRI"NG',
            {
                "bigquery": "STRING",
                "clickhouse": "String",
                "snowflake": "VARCHAR",
                "sqlite": "text",
                "trino": "varchar(7)",
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
                "risingwave": "text",
                "flink": "CHAR(7) NOT NULL",
            },
            id="string-quote2",
            marks=[
                pytest.mark.broken(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-25716",
                ),
                pytest.mark.broken(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason='sql parser error: Expected end of statement, found: "NG\'" at line:1, column:31 Near "SELECT \'STRI"NG\' AS "\'STRI""',
                ),
            ],
        ),
    ],
)
def test_string_literal(con, backend, text_value, expected_types):
    expr = ibis.literal(text_value)
    result = con.execute(expr)
    assert result == text_value

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == expected_types[backend_name]


def is_text_type(x):
    return isinstance(x, str)


def test_string_col_is_unicode(alltypes, df):
    dtype = alltypes.string_col.type()
    assert dtype == dt.String(nullable=dtype.nullable)
    assert df.string_col.map(is_text_type).all()
    result = alltypes.string_col.execute()
    assert result.map(is_text_type).all()


def uses_java_re(t):
    backend_name = t._find_backend().name
    return backend_name in {"pyspark", "flink"}


@pytest.mark.parametrize(
    ("result_func", "expected_func"),
    [
        param(
            lambda t: t.string_col.contains("6"),
            lambda t: t.string_col.str.contains("6"),
            id="contains",
            marks=[
                pytest.mark.broken(
                    ["mssql"], raises=PyODBCProgrammingError, reason="incorrect syntax"
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like("6%"),
            lambda t: t.string_col.str.contains("6.*"),
            id="like",
            marks=[
                pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(
            lambda t: t.string_col.like("6^%"),
            lambda t: t.string_col.str.contains("6%"),
            id="complex_like_escape",
            marks=[
                pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(
            lambda t: t.string_col.like("6^%%"),
            lambda t: t.string_col.str.contains("6%.*"),
            id="complex_like_escape_match",
            marks=[
                pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
            ],
        ),
        param(
            lambda t: t.string_col.ilike("6%"),
            lambda t: t.string_col.str.contains("6.*"),
            id="ilike",
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="mssql doesn't allow like outside of filters",
                    raises=PyODBCProgrammingError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.rlike("|".join(map(str, range(10)))),
            lambda t: t.string_col == t.string_col,
            id="rlike",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: ("a" + t.string_col + "a").re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search_substring",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_search(
                r"\p{Digit}+" if uses_java_re(t) else r"[[:digit:]]+"
            ),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search_posix",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.never(
                    ["druid"],
                    reason="No posix support; regex is interpreted literally",
                    raises=AssertionError,
                ),
            ],
        ),
        param(
            lambda t: ("xyz" + t.string_col + "abcd").re_extract(r"(\d+)", 0),
            lambda t: t.string_col.str.extract(r"(\d+)", expand=False),
            id="re_extract",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: ("xyz" + t.string_col + "abcd").re_extract(r"(\d+)abc", 1),
            lambda t: t.string_col.str.extract(r"(\d+)", expand=False),
            id="re_extract_group",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_extract(
                r"(\p{Digit}+)" if uses_java_re(t) else r"([[:digit:]]+)", 1
            ),
            lambda t: t.string_col.str.extract(r"(\d+)", expand=False),
            id="re_extract_posix",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(
                    ["druid"], reason="No posix support", raises=AssertionError
                ),
            ],
        ),
        param(
            lambda t: (t.string_col + "1").re_extract(r"\d(\d+)", 0),
            lambda t: (t.string_col + "1").str.extract(r"(\d+)", expand=False),
            id="re_extract_whole_group",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 1),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 0],
            id="re_extract_group_1",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 2),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 1],
            id="re_extract_group_2",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 3),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 2],
            id="re_extract_group_3",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"^(\d+)", 1),
            lambda t: t.date_string_col.str.extract(r"^(\d+)", expand=False),
            id="re_extract_group_at_beginning",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)$", 1),
            lambda t: t.date_string_col.str.extract(r"(\d+)$", expand=False),
            id="re_extract_group_at_end",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "exasol"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_replace(
                r"\p{Digit}+" if uses_java_re(t) else r"[[:digit:]]+", "a"
            ),
            lambda t: t.string_col.str.replace(r"\d+", "a", regex=True),
            id="re_replace_posix",
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "mssql", "druid", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_replace(r"\d+", "a"),
            lambda t: t.string_col.str.replace(r"\d+", "a", regex=True),
            id="re_replace",
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "mssql", "druid", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.repeat(2),
            lambda t: t.string_col * 2,
            id="repeat_method",
            marks=pytest.mark.notimpl(
                ["oracle"],
                raises=OracleDatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: 2 * t.string_col,
            lambda t: 2 * t.string_col,
            id="repeat_left",
            marks=pytest.mark.notimpl(
                ["oracle"],
                raises=OracleDatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: t.string_col * 2,
            lambda t: t.string_col * 2,
            id="repeat_right",
            marks=pytest.mark.notimpl(
                ["oracle"],
                raises=OracleDatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: t.string_col.translate("01", "ab"),
            lambda t: t.string_col.str.translate(str.maketrans("01", "ab")),
            id="translate",
            marks=[
                pytest.mark.notimpl(
                    ["mysql", "polars", "druid"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=com.OperationNotDefinedError,
                    reason="doesn't support `TRANSLATE`",
                ),
            ],
        ),
        param(
            lambda t: t.string_col.find("a"),
            lambda t: t.string_col.str.find("a"),
            id="find",
            marks=pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError),
        ),
        param(
            lambda t: t.date_string_col.find("13", 3),
            lambda t: t.date_string_col.str.find("13", 3),
            id="find_start",
            marks=[
                pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t: t.string_col.lpad(10, "a"),
            lambda t: t.string_col.str.pad(10, fillchar="a", side="left"),
            id="lpad",
            marks=pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError),
        ),
        param(
            lambda t: t.string_col.rpad(10, "a"),
            lambda t: t.string_col.str.pad(10, fillchar="a", side="right"),
            id="rpad",
            marks=pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError),
        ),
        param(
            lambda t: t.string_col.find_in_set(["1"]),
            lambda t: t.string_col.str.find("1"),
            id="find_in_set",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "polars",
                        "mssql",
                        "trino",
                        "druid",
                        "oracle",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=com.OperationNotDefinedError,
                    reason="doesn't support `FIND_IN_SET`",
                ),
            ],
        ),
        param(
            lambda t: t.string_col.find_in_set(["a"]),
            lambda t: t.string_col.str.find("a"),
            id="find_in_set_all_missing",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "polars",
                        "mssql",
                        "trino",
                        "druid",
                        "oracle",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=com.OperationNotDefinedError,
                    reason="doesn't support `FIND_IN_SET`",
                ),
            ],
        ),
        param(
            lambda t: t.string_col.lower(),
            lambda t: t.string_col.str.lower(),
            id="lower",
        ),
        param(
            lambda t: t.string_col.upper(),
            lambda t: t.string_col.str.upper(),
            id="upper",
        ),
        param(
            lambda t: t.string_col.reverse(),
            lambda t: t.string_col.str[::-1],
            id="reverse",
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            lambda t: t.string_col.map(ord).astype("int32"),
            id="ascii_str",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "druid"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.length(),
            lambda t: t.string_col.str.len().astype("int32"),
            id="length",
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").startswith(
                "abc"
            ),
            lambda t: t.int_col == 1,
            id="startswith",
            marks=[
                pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").endswith(
                "bcd"
            ),
            lambda t: t.int_col == 1,
            id="endswith",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "mssql"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.startswith("2010-01"),
            lambda t: t.date_string_col.str.startswith("2010-01"),
            id="startswith-simple",
            marks=[
                pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t: t.date_string_col.endswith("/10"),
            lambda t: t.date_string_col.str.endswith("/10"),
            id="endswith-simple",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "mssql"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t: t.string_col.strip(),
            lambda t: t.string_col.str.strip(),
            id="strip",
        ),
        param(
            lambda t: t.string_col.lstrip(),
            lambda t: t.string_col.str.lstrip(),
            id="lstrip",
        ),
        param(
            lambda t: t.string_col.rstrip(),
            lambda t: t.string_col.str.rstrip(),
            id="rstrip",
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:5],
            id="substr",
        ),
        param(
            lambda t: t.date_string_col.substr(2),
            lambda t: t.date_string_col.str[2:],
            id="substr-start-only",
            marks=[
                pytest.mark.broken(
                    ["mssql"],
                    reason="substr requires 3 arguments",
                    raises=PyODBCProgrammingError,
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.left(2),
            lambda t: t.date_string_col.str[:2],
            id="left",
        ),
        param(
            lambda t: t.date_string_col.right(2),
            lambda t: t.date_string_col.str[-2:],
            id="right",
        ),
        param(
            lambda t: t.date_string_col[1:3],
            lambda t: t.date_string_col.str[1:3],
            id="slice",
        ),
        param(
            lambda t: t.date_string_col[-2],
            lambda t: t.date_string_col.str[-2],
            id="negative-index",
            marks=[pytest.mark.broken(["druid"], raises=PyDruidProgrammingError)],
        ),
        param(
            lambda t: t.date_string_col[t.date_string_col.length() - 1 :],
            lambda t: t.date_string_col.str[-1:],
            id="expr_slice_begin",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.xfail_version(datafusion=["datafusion==35"]),
            ],
        ),
        param(
            lambda t: t.date_string_col[: t.date_string_col.length()],
            lambda t: t.date_string_col,
            id="expr_slice_end",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(["druid"], raises=PyDruidProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col[:],
            lambda t: t.date_string_col,
            id="expr_empty_slice",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument "
                        "Subtract(StringLength(date_string_col), 0)"
                    ),
                ),
                pytest.mark.broken(["druid"], raises=PyDruidProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col[
                t.date_string_col.length() - 2 : t.date_string_col.length() - 1
            ],
            lambda t: t.date_string_col.str[-2:-1],
            id="expr_slice_begin_end",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(["druid"], raises=PyDruidProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col.split("/"),
            lambda t: t.date_string_col.str.split("/"),
            id="split",
            marks=pytest.mark.notimpl(
                [
                    "impala",
                    "mysql",
                    "sqlite",
                    "mssql",
                    "druid",
                    "oracle",
                    "flink",
                    "exasol",
                ],
                raises=com.OperationNotDefinedError,
            ),
        ),
        param(
            lambda t: ibis.literal("-").join(["a", t.string_col, "c"]),
            lambda t: "a-" + t.string_col + "-c",
            id="join",
            marks=pytest.mark.notimpl(
                ["exasol"],
                raises=com.OperationNotDefinedError,
            ),
        ),
        param(
            lambda t: t.string_col + t.date_string_col,
            lambda t: t.string_col + t.date_string_col,
            id="concat_columns",
        ),
        param(
            lambda t: t.string_col + "a",
            lambda t: t.string_col + "a",
            id="concat_column_scalar",
        ),
        param(
            lambda t: "a" + t.string_col,
            lambda t: "a" + t.string_col,
            id="concat_scalar_column",
        ),
        param(
            lambda t: t.string_col.replace("1", "42"),
            lambda t: t.string_col.str.replace("1", "42"),
            id="replace",
        ),
    ],
)
def test_string(backend, alltypes, df, result_func, expected_func):
    expr = result_func(alltypes).name("tmp")
    result = expr.execute()

    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(
    ["mysql", "mssql", "druid", "exasol"], raises=com.OperationNotDefinedError
)
def test_re_replace_global(con):
    expr = ibis.literal("aba").re_replace("a", "c")
    result = con.execute(expr)
    assert result == "cbc"


@pytest.mark.notimpl(["druid"], raises=ValidationError)
def test_substr_with_null_values(backend, alltypes, df):
    table = alltypes.mutate(
        substr_col_null=ibis.case()
        .when(alltypes["bool_col"], alltypes["string_col"])
        .else_(None)
        .end()
        .substr(0, 2)
    )
    result = table.execute()

    expected = df.copy()
    mask = ~expected["bool_col"]
    expected["substr_col_null"] = expected["string_col"]
    expected.loc[mask, "substr_col_null"] = None
    expected["substr_col_null"] = expected["substr_col_null"].str.slice(0, 2)

    backend.assert_frame_equal(result.fillna(pd.NA), expected.fillna(pd.NA))


@pytest.mark.parametrize(
    ("result_func", "expected"),
    [
        param(lambda d: d.protocol(), "http", id="protocol"),
        param(
            lambda d: d.authority(),
            "user:pass@example.com:80",
            id="authority",
            marks=[pytest.mark.notyet(["trino"], raises=com.OperationNotDefinedError)],
        ),
        param(
            lambda d: d.userinfo(),
            "user:pass",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse", "snowflake", "trino"],
                    raises=com.OperationNotDefinedError,
                    reason="doesn't support `USERINFO`",
                )
            ],
            id="userinfo",
        ),
        param(
            lambda d: d.host(),
            "example.com",
            id="host",
            marks=[
                pytest.mark.notyet(
                    ["snowflake"],
                    raises=com.OperationNotDefinedError,
                    reason="host is netloc",
                ),
                pytest.mark.broken(
                    ["clickhouse"],
                    reason="Backend is foiled by the presence of a password",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda d: d.file(),
            "/docs/books/tutorial/index.html?name=networking",
            id="file",
            marks=[
                pytest.mark.notimpl(
                    ["pandas", "dask", "datafusion", "sqlite"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(lambda d: d.path(), "/docs/books/tutorial/index.html", id="path"),
        param(lambda d: d.query(), "name=networking", id="query"),
        param(lambda d: d.query("name"), "networking", id="query-key"),
        param(
            lambda d: d.query(ibis.literal("na") + ibis.literal("me")),
            "networking",
            id="query-dynamic-key",
        ),
        param(lambda d: d.fragment(), "DOWNLOADING", id="ref"),
    ],
)
@pytest.mark.notimpl(
    [
        "bigquery",
        "duckdb",
        "exasol",
        "mssql",
        "mysql",
        "polars",
        "postgres",
        "risingwave",
        "pyspark",
        "druid",
        "oracle",
    ],
    raises=com.OperationNotDefinedError,
)
def test_parse_url(con, result_func, expected):
    url = "http://user:pass@example.com:80/docs/books/tutorial/index.html?name=networking#DOWNLOADING"
    expr = result_func(ibis.literal(url))
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("inp, expected"),
    [
        param(
            None,
            None,
            id="none",
            marks=[
                pytest.mark.notyet(
                    ["druid"],
                    raises=PyDruidProgrammingError,
                    reason="illegal use of NULL",
                )
            ],
        ),
        param(
            "",
            "",
            id="empty",
            marks=[
                pytest.mark.notyet(
                    ["oracle"],
                    reason="https://github.com/oracle/python-oracledb/issues/298",
                    raises=AssertionError,
                ),
                pytest.mark.notyet(["exasol"], raises=AssertionError),
            ],
        ),
        param("Abc", "Abc", id="no_change"),
        param("abc", "Abc", id="lower_to_upper"),
        param("aBC", "Abc", id="mixed_to_upper"),
        param(" abc", " abc", id="leading_space"),
        param("9abc", "9abc", id="leading_digit"),
        param("aBc dEf", "Abc def", id="mixed_with_space"),
        param("aBc-dEf", "Abc-def", id="mixed_with_hyphen"),
        param("aBc1dEf", "Abc1def", id="mixed_with_digit"),
    ],
)
def test_capitalize(con, inp, expected):
    s = ibis.literal(inp, type="string")
    expr = s.capitalize()
    result = con.execute(expr)
    if expected is not None:
        assert result == expected
    else:
        assert pd.isnull(result)


@pytest.mark.notimpl(
    [
        "dask",
        "pandas",
        "polars",
        "oracle",
        "flink",
        "sqlite",
        "mssql",
        "mysql",
        "exasol",
        "impala",
    ],
    raises=com.OperationNotDefinedError,
)
def test_array_string_join(con):
    s = ibis.array(["a", "b", "c"])
    expected = "a,b,c"
    expr = ibis.literal(",").join(s)
    assert con.execute(expr) == expected

    expr = s.join(",")
    assert con.execute(expr) == expected


@pytest.mark.notimpl(
    ["mssql", "mysql", "druid", "exasol"], raises=com.OperationNotDefinedError
)
def test_subs_with_re_replace(con):
    expr = ibis.literal("hi").re_replace("i", "a").substitute({"d": "b"}, else_="k")
    result = con.execute(expr)
    assert result == "k"


def test_multiple_subs(con):
    m = {"foo": "FOO", "bar": "BAR"}
    expr = ibis.literal("foo").substitute(m)
    result = con.execute(expr)
    assert result == "FOO"


@pytest.mark.notimpl(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "druid",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "sqlite",
        "flink",
        "exasol",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function levenshtein(character varying, character varying) does not exist",
)
@pytest.mark.parametrize(
    "right", ["sitting", ibis.literal("sitting")], ids=["python", "ibis"]
)
def test_levenshtein(con, right):
    left = ibis.literal("kitten")
    expr = left.levenshtein(right)
    result = con.execute(expr)
    assert result == 3


@pytest.mark.parametrize(
    "expr",
    [
        param(ibis.case().when(True, "%").end(), id="case"),
        param(ibis.ifelse(True, "%", ibis.NA), id="ifelse"),
    ],
)
def test_no_conditional_percent_escape(con, expr):
    assert con.execute(expr) == "%"


@pytest.mark.notimpl(["mssql", "exasol"], raises=com.OperationNotDefinedError)
def test_non_match_regex_search_is_false(con):
    expr = ibis.literal("foo").re_search("bar")
    result = con.execute(expr)
    assert isinstance(result, (bool, np.bool_))
    assert not result


@pytest.mark.notimpl(
    [
        "dask",
        "impala",
        "mysql",
        "sqlite",
        "mssql",
        "druid",
        "oracle",
        "flink",
        "exasol",
        "pandas",
        "bigquery",
    ],
    raises=com.OperationNotDefinedError,
)
def test_re_split(con):
    lit = ibis.literal(",a,,,,c")
    expr = lit.re_split(",+")
    result = con.execute(expr)
    assert list(result) == ["", "a", "c"]


@pytest.mark.notimpl(
    [
        "dask",
        "impala",
        "mysql",
        "sqlite",
        "mssql",
        "druid",
        "oracle",
        "flink",
        "exasol",
        "pandas",
        "bigquery",
    ],
    raises=com.OperationNotDefinedError,
)
def test_re_split_column(alltypes):
    expr = alltypes.limit(5).string_col.re_split(r"\d+")
    result = expr.execute()
    assert all(not any(element) for element in result)


@pytest.mark.notimpl(
    [
        "dask",
        "impala",
        "mysql",
        "sqlite",
        "mssql",
        "druid",
        "oracle",
        "flink",
        "exasol",
        "pandas",
        "bigquery",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="clickhouse only supports pattern constants",
)
@pytest.mark.notyet(
    ["polars"],
    raises=BaseException,  # yikes, panic exception
    reason="pyarrow doesn't support splitting on a pattern per row",
)
@pytest.mark.notyet(
    ["datafusion"],
    raises=Exception,
    reason="pyarrow doesn't support splitting on a pattern per row",
)
def test_re_split_column_multiple_patterns(alltypes):
    expr = (
        alltypes.filter(lambda t: t.string_col.isin(("1", "2")))
        .select(
            splits=lambda t: t.string_col.re_split(
                ibis.ifelse(t.string_col == "1", "0|1", r"\d+")
            )
        )
        .splits
    )
    result = expr.execute()
    assert all(not any(element) for element in result)


@pytest.mark.parametrize(
    "fn",
    [lambda n: n + "a", lambda n: n + n, lambda n: "a" + n],
    ids=["null-a", "null-null", "a-null"],
)
@pytest.mark.notimpl(["pandas", "dask"], raises=TypeError)
def test_concat_with_null(con, fn):
    null = ibis.literal(None, type="string")
    expr = fn(null)
    result = con.execute(expr)
    assert pd.isna(result)


@pytest.mark.parametrize(
    "args",
    [
        param((ibis.literal(None, str), None), id="null-null"),
        param((ibis.literal("abc"), None), id="abc-null"),
        param((ibis.literal("abc"), ibis.literal(None, str)), id="abc-typed-null"),
        param((ibis.literal("abc"), "def", None), id="abc-def-null"),
    ],
)
@pytest.mark.parametrize(
    "method",
    [lambda args: args[0].concat(*args[1:]), lambda args: reduce(add, args)],
    ids=["concat", "add"],
)
@pytest.mark.notimpl(["pandas", "dask"], raises=TypeError)
def test_concat(con, args, method):
    expr = method(args)
    assert pd.isna(con.execute(expr))
