from __future__ import annotations

import contextlib

import pandas as pd
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import OperationNotDefinedError

try:
    from pyspark.sql.utils import PythonException
except ImportError:
    PythonException = None

try:
    from google.api_core.exceptions import BadRequest
except ImportError:
    BadRequest = None


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
                "flink": "CHAR(7) NOT NULL",
            },
            id="string-quote1",
            marks=pytest.mark.broken(
                ["oracle"],
                raises=sa.exc.DatabaseError,
                reason="ORA-01741: illegal zero length identifier",
            ),
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
                "flink": "CHAR(7) NOT NULL",
            },
            id="string-quote2",
            marks=pytest.mark.broken(
                ["oracle"],
                raises=sa.exc.DatabaseError,
                reason="ORA-25716",
            ),
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
                    ["mssql"],
                    raises=sa.exc.ProgrammingError,
                    reason=(
                        "(pymssql._pymssql.ProgrammingError) (102, b\"Incorrect syntax near '>'."
                        "DB-Lib error message 20018, severity 15:\nGeneral SQL Server error: "
                        'Check messages from the SQL Server\n")'
                        "[SQL: SELECT charindex(%(param_1)s, t0.string_col) - %(charindex_1)s >= "
                        "%(param_2)s AS tmp"
                        "FROM functional_alltypes AS t0]"
                    ),
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like("6%"),
            lambda t: t.string_col.str.contains("6.*"),
            id="like",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "polars"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="mssql doesn't allow like outside of filters",
                    raises=sa.exc.OperationalError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like("6^%"),
            lambda t: t.string_col.str.contains("6%"),
            id="complex_like_escape",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "polars"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="mssql doesn't allow like outside of filters",
                    raises=sa.exc.OperationalError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like("6^%%"),
            lambda t: t.string_col.str.contains("6%.*"),
            id="complex_like_escape_match",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "polars"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="mssql doesn't allow like outside of filters",
                    raises=sa.exc.OperationalError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.ilike("6%"),
            lambda t: t.string_col.str.contains("6.*"),
            id="ilike",
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "pyspark", "polars"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="mssql doesn't allow like outside of filters",
                    raises=sa.exc.OperationalError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.rlike("|".join(map(str, range(10)))),
            lambda t: t.string_col == t.string_col,
            id="rlike",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: ("a" + t.string_col + "a").re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search_substring",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mssql", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(["pyspark"], raises=PythonException),
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
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: ("xyz" + t.string_col + "abcd").re_extract(r"(\d+)abc", 1),
            lambda t: t.string_col.str.extract(r"(\d+)", expand=False),
            id="re_extract_group",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: (t.string_col + "1").re_extract(r"\d(\d+)", 0),
            lambda t: (t.string_col + "1").str.extract(r"(\d+)", expand=False),
            id="re_extract_whole_group",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"^(\d+)", 1),
            lambda t: t.date_string_col.str.extract(r"^(\d+)", expand=False),
            id="re_extract_group_at_beginning",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)$", 1),
            lambda t: t.date_string_col.str.extract(r"(\d+)$", expand=False),
            id="re_extract_group_at_end",
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "druid", "oracle"],
                    raises=com.OperationNotDefinedError,
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
                    ["mysql", "mssql", "druid", "oracle"],
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
                    ["mysql", "mssql", "druid", "oracle"],
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
                raises=sa.exc.DatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: 2 * t.string_col,
            lambda t: 2 * t.string_col,
            id="repeat_left",
            marks=pytest.mark.notimpl(
                ["oracle"],
                raises=sa.exc.DatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: t.string_col * 2,
            lambda t: t.string_col * 2,
            id="repeat_right",
            marks=pytest.mark.notimpl(
                ["oracle"],
                raises=sa.exc.DatabaseError,
                reason="ORA-00904: REPEAT invalid identifier",
            ),
        ),
        param(
            lambda t: t.string_col.translate("01", "ab"),
            lambda t: t.string_col.str.translate(str.maketrans("01", "ab")),
            id="translate",
            marks=[
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "duckdb",
                        "mssql",
                        "mysql",
                        "polars",
                        "druid",
                        "oracle",
                    ],
                    raises=com.OperationNotDefinedError,
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
                pytest.mark.notyet(["bigquery"], raises=NotImplementedError),
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
            marks=pytest.mark.notimpl(
                [
                    "bigquery",
                    "datafusion",
                    "pyspark",
                    "sqlite",
                    "snowflake",
                    "polars",
                    "mssql",
                    "trino",
                    "druid",
                    "oracle",
                ],
                raises=com.OperationNotDefinedError,
            ),
        ),
        param(
            lambda t: t.string_col.find_in_set(["a"]),
            lambda t: t.string_col.str.find("a"),
            id="find_in_set_all_missing",
            marks=pytest.mark.notimpl(
                [
                    "bigquery",
                    "datafusion",
                    "pyspark",
                    "sqlite",
                    "snowflake",
                    "polars",
                    "mssql",
                    "trino",
                    "druid",
                    "oracle",
                ],
                raises=com.OperationNotDefinedError,
            ),
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
            # pyspark doesn't support `cases` yet
            marks=[
                pytest.mark.notimpl(
                    ["dask", "pyspark"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["druid"],
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    raises=sa.exc.OperationalError,
                    reason=(
                        '(pymssql._pymssql.OperationalError) (156, b"Incorrect syntax near the keyword '
                        "'LIKE'.DB-Lib error message 20018, severity 15:\nGeneral SQL Server error: "
                        'Check messages from the SQL Server\n")'
                        "[SQL: SELECT (CASE t0.int_col WHEN %(param_1)s THEN %(param_2)s WHEN %(param_3)s "
                        "THEN %(param_4)s ELSE %(param_5)s END LIKE %(param_6)s + '%') AS tmp"
                        "FROM functional_alltypes AS t0]"
                    ),
                ),
            ],
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").endswith(
                "bcd"
            ),
            lambda t: t.int_col == 1,
            id="endswith",
            # pyspark doesn't support `cases` yet
            marks=[
                pytest.mark.notimpl(
                    ["dask", "datafusion", "pyspark"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
                pytest.mark.broken(
                    ["mssql"],
                    reason=(
                        '(pymssql._pymssql.OperationalError) (156, b"Incorrect syntax near '
                        "the keyword 'LIKE'.DB-Lib error message 20018, severity 15:\n"
                        'General SQL Server error: Check messages from the SQL Server\n")'
                    ),
                    raises=sa.exc.OperationalError,
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.startswith("2010-01"),
            lambda t: t.date_string_col.str.startswith("2010-01"),
            id="startswith-simple",
            marks=[
                pytest.mark.notimpl(
                    ["dask"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    raises=sa.exc.OperationalError,
                    reason=(
                        '(pymssql._pymssql.OperationalError) (156, b"Incorrect syntax near '
                        "the keyword 'LIKE'.DB-Lib error message 20018, severity 15:\n"
                        'General SQL Server error: Check messages from the SQL Server\n")'
                    ),
                ),
            ],
        ),
        param(
            lambda t: t.date_string_col.endswith("/10"),
            lambda t: t.date_string_col.str.endswith("/10"),
            id="endswith-simple",
            marks=[
                pytest.mark.notimpl(
                    ["dask", "datafusion"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["mssql"],
                    raises=sa.exc.OperationalError,
                    reason=(
                        '(pymssql._pymssql.OperationalError) (156, b"Incorrect syntax near '
                        "the keyword 'LIKE'.DB-Lib error message 20018, severity 15:\n"
                        'General SQL Server error: Check messages from the SQL Server\n")'
                    ),
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
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id="capitalize",
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
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["polars"],
                    raises=AttributeError,
                    reason="'NoneType' object has no attribute 'name'",
                ),
                pytest.mark.broken(
                    ["mssql"],
                    reason="substr requires 3 arguments",
                    raises=sa.exc.OperationalError,
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
            marks=[
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
                pytest.mark.broken(["datafusion", "impala"], raises=AssertionError),
                pytest.mark.notimpl(["pyspark"], raises=NotImplementedError),
            ],
        ),
        param(
            lambda t: t.date_string_col[t.date_string_col.length() - 1 :],
            lambda t: t.date_string_col.str[-1:],
            id="expr_slice_begin",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=NotImplementedError,
                    reason=(
                        "Specifying `start` or `length` with column expressions is not supported."
                    ),
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(
                    ["dask"],
                    reason="'Series' object has no attribute 'items'",
                    raises=AttributeError,
                ),
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col[: t.date_string_col.length()],
            lambda t: t.date_string_col,
            id="expr_slice_end",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=NotImplementedError,
                    reason=(
                        "Specifying `start` or `length` with column expressions is not supported."
                    ),
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(
                    ["dask"],
                    reason="'Series' object has no attribute 'items'",
                    raises=AttributeError,
                ),
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col[:],
            lambda t: t.date_string_col,
            id="expr_empty_slice",
            # TODO: substring #2553
            marks=[
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=NotImplementedError,
                    reason=(
                        "Specifying `start` or `length` with column expressions is not supported."
                    ),
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument "
                        "Subtract(StringLength(date_string_col), 0)"
                    ),
                ),
                pytest.mark.broken(
                    ["dask"],
                    reason="'Series' object has no attribute 'items'",
                    raises=AttributeError,
                ),
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
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
                    ["pyspark"],
                    raises=NotImplementedError,
                    reason=(
                        "Specifying `start` or `length` with column expressions is not supported."
                    ),
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason=(
                        "Polars does not support columnar argument Subtract(StringLength(date_string_col), 1)"
                    ),
                ),
                pytest.mark.broken(
                    ["dask"],
                    reason="'Series' object has no attribute 'items'",
                    raises=AttributeError,
                ),
                pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError),
            ],
        ),
        param(
            lambda t: t.date_string_col.split("/"),
            lambda t: t.date_string_col.str.split("/"),
            id="split",
            marks=pytest.mark.notimpl(
                [
                    "dask",
                    "datafusion",
                    "impala",
                    "mysql",
                    "sqlite",
                    "mssql",
                    "druid",
                    "oracle",
                ],
                raises=com.OperationNotDefinedError,
            ),
        ),
        param(
            lambda t: ibis.literal("-").join(["a", t.string_col, "c"]),
            lambda t: "a-" + t.string_col + "-c",
            id="join",
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
    ["mysql", "mssql", "druid", "oracle"],
    raises=com.OperationNotDefinedError,
)
def test_re_replace_global(con):
    expr = ibis.literal("aba").re_replace("a", "c")
    result = con.execute(expr)
    assert result == "cbc"


@pytest.mark.broken(
    ["mssql"],
    raises=sa.exc.OperationalError,
    reason=(
        '(pymssql._pymssql.OperationalError) (4145, b"An expression of non-boolean type specified in '
        "a context where a condition is expected, near 'THEN'.DB-Lib error message 20018, severity 15:\n"
    ),
)
@pytest.mark.notimpl(["druid"], raises=ValidationError)
@pytest.mark.broken(
    ["oracle"],
    raises=sa.exc.DatabaseError,
    reason="ORA-61801: only boolean column or attribute can be used as a predicate",
)
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
                    raises=OperationNotDefinedError,
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
                    raises=OperationNotDefinedError,
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
        "mssql",
        "mysql",
        "polars",
        "postgres",
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


def test_capitalize(con):
    s = ibis.literal("aBc")
    expected = "Abc"
    expr = s.capitalize()
    assert con.execute(expr) == expected


@pytest.mark.notimpl(
    ["dask", "datafusion", "pandas", "polars", "druid", "oracle"],
    raises=OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["impala", "mssql", "mysql", "sqlite"],
    reason="no arrays",
    raises=OperationNotDefinedError,
)
def test_array_string_join(con):
    s = ibis.array(["a", "b", "c"])
    expected = "a,b,c"
    expr = ibis.literal(",").join(s)
    assert con.execute(expr) == expected

    expr = s.join(",")
    assert con.execute(expr) == expected


@pytest.mark.notimpl(
    ["mssql", "mysql", "pyspark", "druid", "oracle"],
    raises=com.OperationNotDefinedError,
)
def test_subs_with_re_replace(con):
    expr = ibis.literal("hi").re_replace("i", "a").substitute({"d": "b"}, else_="k")
    result = con.execute(expr)
    assert result == "k"


@pytest.mark.notimpl(["pyspark"], raises=com.OperationNotDefinedError)
def test_multiple_subs(con):
    m = {"foo": "FOO", "bar": "BAR"}
    expr = ibis.literal("foo").substitute(m)
    result = con.execute(expr)
    assert result == "FOO"


@pytest.mark.notimpl(
    [
        "bigquery",
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
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize(
    "right", ["sitting", ibis.literal("sitting")], ids=["python", "ibis"]
)
def test_levenshtein(con, right):
    left = ibis.literal("kitten")
    expr = left.levenshtein(right)
    result = con.execute(expr)
    assert result == 3


@pytest.mark.notyet(
    ["mssql"],
    reason="doesn't allow boolean expressions in select statements",
    raises=sa.exc.OperationalError,
)
@pytest.mark.broken(
    ["oracle"],
    reason="sqlalchemy converts True to 1, which cannot be used in CASE WHEN statement",
    raises=sa.exc.DatabaseError,
)
@pytest.mark.parametrize(
    "expr",
    [
        param(ibis.case().when(True, "%").end(), id="case"),
        param(ibis.ifelse(True, "%", ibis.NA), id="ifelse"),
    ],
)
def test_no_conditional_percent_escape(con, expr):
    assert con.execute(expr) == "%"
