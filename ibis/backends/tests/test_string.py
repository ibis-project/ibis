import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.common.exceptions import OperationNotDefinedError


def is_text_type(x):
    return isinstance(x, str)


def test_string_col_is_unicode(alltypes, df):
    dtype = alltypes.string_col.type()
    assert dtype == dt.String(nullable=dtype.nullable)
    assert df.string_col.map(is_text_type).all()
    result = alltypes.string_col.execute()
    assert result.map(is_text_type).all()


@pytest.mark.parametrize(
    ('result_func', 'expected_func'),
    [
        param(
            lambda t: t.string_col.contains('6'),
            lambda t: t.string_col.str.contains('6'),
            id='contains',
            marks=pytest.mark.notimpl(["datafusion", "mssql"]),
        ),
        param(
            lambda t: t.string_col.like('6%'),
            lambda t: t.string_col.str.contains('6.*'),
            id='like',
            marks=[
                pytest.mark.notimpl(["datafusion", "polars"]),
                pytest.mark.notyet(
                    ["mssql"], reason="mssql doesn't allow like outside of filters"
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like('6^%'),
            lambda t: t.string_col.str.contains('6%'),
            id='complex_like_escape',
            marks=[
                pytest.mark.notimpl(["datafusion", "polars"]),
                pytest.mark.notyet(
                    ["mssql"], reason="mssql doesn't allow like outside of filters"
                ),
            ],
        ),
        param(
            lambda t: t.string_col.like('6^%%'),
            lambda t: t.string_col.str.contains('6%.*'),
            id='complex_like_escape_match',
            marks=[
                pytest.mark.notimpl(["datafusion", "polars"]),
                pytest.mark.notyet(
                    ["mssql"], reason="mssql doesn't allow like outside of filters"
                ),
            ],
        ),
        param(
            lambda t: t.string_col.ilike('6%'),
            lambda t: t.string_col.str.contains('6.*'),
            id='ilike',
            marks=[
                pytest.mark.notimpl(["datafusion", "pyspark", "polars"]),
                pytest.mark.notyet(
                    ["mssql"], reason="mssql doesn't allow like outside of filters"
                ),
            ],
        ),
        param(
            lambda t: t.string_col.re_search(r'\d+'),
            lambda t: t.string_col.str.contains(r'\d+'),
            id='re_search',
            marks=pytest.mark.notimpl(["impala", "datafusion", "mssql"]),
        ),
        param(
            lambda t: t.string_col.re_search(r'[[:digit:]]+'),
            lambda t: t.string_col.str.contains(r'\d+'),
            id='re_search_posix',
            marks=pytest.mark.notimpl(["datafusion", "pyspark", "mssql"]),
        ),
        param(
            lambda t: t.string_col.re_extract(r'(\d+)', 1),
            lambda t: t.string_col.str.extract(r'(\d+)', expand=False),
            id='re_extract',
            marks=pytest.mark.notimpl(["impala", "mysql", "mssql"]),
        ),
        param(
            lambda t: t.string_col.re_extract(r'([[:digit:]]+)', 1),
            lambda t: t.string_col.str.extract(r'(\d+)', expand=False),
            id='re_extract_posix',
            marks=pytest.mark.notimpl(["mysql", "pyspark", "mssql"]),
        ),
        param(
            lambda t: (t.string_col + "1").re_extract(r'\d(\d+)', 0),
            lambda t: (t.string_col + "1").str.extract(r'(\d+)', expand=False),
            id='re_extract_whole_group',
            marks=pytest.mark.notimpl(["impala", "mysql", "snowflake", "mssql"]),
        ),
        param(
            lambda t: t.string_col.re_replace(r'[[:digit:]]+', 'a'),
            lambda t: t.string_col.str.replace(r'\d+', 'a', regex=True),
            id='re_replace_posix',
            marks=pytest.mark.notimpl(['datafusion', "mysql", "pyspark", "mssql"]),
        ),
        param(
            lambda t: t.string_col.re_replace(r'\d+', 'a'),
            lambda t: t.string_col.str.replace(r'\d+', 'a', regex=True),
            id='re_replace',
            marks=pytest.mark.notimpl(["impala", "datafusion", "mysql", "mssql"]),
        ),
        param(
            lambda t: t.string_col.repeat(2),
            lambda t: t.string_col * 2,
            id="repeat_method",
        ),
        param(lambda t: 2 * t.string_col, lambda t: 2 * t.string_col, id="repeat_left"),
        param(
            lambda t: t.string_col * 2, lambda t: t.string_col * 2, id="repeat_right"
        ),
        param(
            lambda t: t.string_col.translate('01', 'ab'),
            lambda t: t.string_col.str.translate(str.maketrans('01', 'ab')),
            id='translate',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "datafusion",
                    "duckdb",
                    "mssql",
                    "mysql",
                    "polars",
                ]
            ),
        ),
        param(
            lambda t: t.string_col.find('a'),
            lambda t: t.string_col.str.find('a'),
            id='find',
            marks=pytest.mark.notimpl(["datafusion", "polars"]),
        ),
        param(
            lambda t: t.string_col.lpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='left'),
            id='lpad',
            marks=pytest.mark.notimpl(["mssql"]),
        ),
        param(
            lambda t: t.string_col.rpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='right'),
            id='rpad',
            marks=pytest.mark.notimpl(["mssql"]),
        ),
        param(
            lambda t: t.string_col.find_in_set(['1']),
            lambda t: t.string_col.str.find('1'),
            id='find_in_set',
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
                ]
            ),
        ),
        param(
            lambda t: t.string_col.find_in_set(['a']),
            lambda t: t.string_col.str.find('a'),
            id='find_in_set_all_missing',
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
                ]
            ),
        ),
        param(
            lambda t: t.string_col.lower(),
            lambda t: t.string_col.str.lower(),
            id='lower',
        ),
        param(
            lambda t: t.string_col.upper(),
            lambda t: t.string_col.str.upper(),
            id='upper',
        ),
        param(
            lambda t: t.string_col.reverse(),
            lambda t: t.string_col.str[::-1],
            id='reverse',
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            lambda t: t.string_col.map(ord).astype('int32'),
            id='ascii_str',
            marks=pytest.mark.notimpl(["clickhouse", "datafusion", "polars"]),
        ),
        param(
            lambda t: t.string_col.length(),
            lambda t: t.string_col.str.len().astype('int32'),
            id='length',
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").startswith(
                "abc"
            ),
            lambda t: t.int_col == 1,
            id='startswith',
            # pyspark doesn't support `cases` yet
            marks=pytest.mark.notimpl(
                ["dask", "datafusion", "pyspark", "pandas", "mssql"]
            ),
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").endswith(
                "bcd"
            ),
            lambda t: t.int_col == 1,
            id='endswith',
            # pyspark doesn't support `cases` yet
            marks=pytest.mark.notimpl(
                ["dask", "datafusion", "pyspark", "pandas", "mssql"]
            ),
        ),
        param(
            lambda t: t.date_string_col.startswith("2010-01"),
            lambda t: t.date_string_col.str.startswith("2010-01"),
            id='startswith-simple',
            marks=pytest.mark.notimpl(["dask", "datafusion", "pandas", "mssql"]),
        ),
        param(
            lambda t: t.date_string_col.endswith("100"),
            lambda t: t.date_string_col.str.endswith("100"),
            id='endswith-simple',
            marks=pytest.mark.notimpl(["dask", "datafusion", "pandas", "mssql"]),
        ),
        param(
            lambda t: t.string_col.strip(),
            lambda t: t.string_col.str.strip(),
            id='strip',
        ),
        param(
            lambda t: t.string_col.lstrip(),
            lambda t: t.string_col.str.lstrip(),
            id='lstrip',
        ),
        param(
            lambda t: t.string_col.rstrip(),
            lambda t: t.string_col.str.rstrip(),
            id='rstrip',
        ),
        param(
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id='capitalize',
            marks=pytest.mark.notimpl(["clickhouse", "duckdb", "mssql"]),
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:5],
            id='substr',
        ),
        param(
            lambda t: t.date_string_col.substr(2),
            lambda t: t.date_string_col.str[2:],
            id='substr-start-only',
            marks=[
                pytest.mark.notimpl(["datafusion", "polars", "pyspark"]),
                pytest.mark.notyet(["mssql"], reason="substr requires 3 arguments"),
            ],
        ),
        param(
            lambda t: t.date_string_col.left(2),
            lambda t: t.date_string_col.str[:2],
            id='left',
        ),
        param(
            lambda t: t.date_string_col.right(2),
            lambda t: t.date_string_col.str[-2:],
            id="right",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            lambda t: t.date_string_col[1:3],
            lambda t: t.date_string_col.str[1:3],
            id='slice',
        ),
        param(
            lambda t: t.date_string_col[t.date_string_col.length() - 1 :],
            lambda t: t.date_string_col.str[-1:],
            id='expr_slice_begin',
            # TODO: substring #2553
            marks=pytest.mark.notimpl(["dask", "pyspark", "polars"]),
        ),
        param(
            lambda t: t.date_string_col[: t.date_string_col.length()],
            lambda t: t.date_string_col,
            id='expr_slice_end',
            # TODO: substring #2553
            marks=pytest.mark.notimpl(["dask", "pyspark", "polars"]),
        ),
        param(
            lambda t: t.date_string_col[:],
            lambda t: t.date_string_col,
            id='expr_empty_slice',
            # TODO: substring #2553
            marks=pytest.mark.notimpl(["dask", "pyspark", "polars"]),
        ),
        param(
            lambda t: t.date_string_col[
                t.date_string_col.length() - 2 : t.date_string_col.length() - 1
            ],
            lambda t: t.date_string_col.str[-2:-1],
            id='expr_slice_begin_end',
            # TODO: substring #2553
            marks=pytest.mark.notimpl(["dask", "pyspark", "polars"]),
        ),
        param(
            lambda t: t.date_string_col.split('/'),
            lambda t: t.date_string_col.str.split('/'),
            id='split',
            marks=pytest.mark.notimpl(
                ["dask", "datafusion", "impala", "mysql", "sqlite", "mssql"]
            ),
        ),
        param(
            lambda t: ibis.literal('-').join(['a', t.string_col, 'c']),
            lambda t: 'a-' + t.string_col + '-c',
            id='join',
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            lambda t: t.string_col + t.date_string_col,
            lambda t: t.string_col + t.date_string_col,
            id='concat_columns',
        ),
        param(
            lambda t: t.string_col + 'a',
            lambda t: t.string_col + 'a',
            id='concat_column_scalar',
        ),
        param(
            lambda t: 'a' + t.string_col,
            lambda t: 'a' + t.string_col,
            id='concat_scalar_column',
        ),
        param(
            lambda t: t.string_col.replace("1", "42"),
            lambda t: t.string_col.str.replace("1", "42"),
            id="replace",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
    ],
)
def test_string(backend, alltypes, df, result_func, expected_func):
    expr = result_func(alltypes).name('tmp')
    result = expr.execute()

    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["datafusion", "mysql", "mssql"])
def test_re_replace_global(con):
    expr = ibis.literal("aba").re_replace("a", "c")
    result = con.execute(expr)
    assert result == "cbc"


@pytest.mark.notimpl(["datafusion", "mssql"])
def test_substr_with_null_values(backend, alltypes, df):
    table = alltypes.mutate(
        substr_col_null=ibis.case()
        .when(alltypes['bool_col'], alltypes['string_col'])
        .else_(None)
        .end()
        .substr(0, 2)
    )
    result = table.execute()

    expected = df.copy()
    mask = ~expected['bool_col']
    expected['substr_col_null'] = expected['string_col']
    expected.loc[mask, 'substr_col_null'] = None
    expected['substr_col_null'] = expected['substr_col_null'].str.slice(0, 2)

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("result_func", "expected"),
    [
        param(lambda d: d.protocol(), "http", id="protocol"),
        param(
            lambda d: d.authority(),
            "user:pass@example.com:80",
            id="authority",
            marks=[pytest.mark.notyet(["trino"])],
        ),
        param(
            lambda d: d.userinfo(),
            "user:pass",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse", "snowflake", "trino"],
                    raises=(NotImplementedError, OperationNotDefinedError),
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
                pytest.mark.broken(
                    ["clickhouse"],
                    reason="Backend is foiled by the presence of a password",
                ),
                pytest.mark.notyet(
                    ["snowflake"],
                    raises=(NotImplementedError, OperationNotDefinedError),
                    reason="host is netloc",
                ),
            ],
        ),
        param(
            lambda d: d.file(),
            "/docs/books/tutorial/index.html?name=networking",
            id="file",
        ),
        param(lambda d: d.path(), "/docs/books/tutorial/index.html", id="path"),
        param(lambda d: d.query(), "name=networking", id="query"),
        param(lambda d: d.query('name'), "networking", id="query-key"),
        param(
            lambda d: d.query(ibis.literal('na') + ibis.literal('me')),
            "networking",
            id="query-dynamic-key",
        ),
        param(lambda d: d.fragment(), "DOWNLOADING", id="ref"),
    ],
)
@pytest.mark.notimpl(
    [
        "bigquery",
        "dask",
        "datafusion",
        "duckdb",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "sqlite",
    ]
)
def test_parse_url(con, result_func, expected):
    url = "http://user:pass@example.com:80/docs/books/tutorial/index.html?name=networking#DOWNLOADING"
    expr = result_func(ibis.literal(url).name("url"))
    result = con.execute(expr)
    assert result == expected
