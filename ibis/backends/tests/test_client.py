import platform
import re

import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from pytest import mark, param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.util import guid


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


def _create_temp_table_with_schema(con, temp_table_name, schema, data=None):
    con.drop_table(temp_table_name, force=True)
    con.create_table(temp_table_name, schema=schema)
    temporary = con.table(temp_table_name)
    assert temporary.execute().empty

    if data is not None and isinstance(data, pd.DataFrame):
        con.load_data(temp_table_name, data, if_exists="append")
        result = temporary.execute()
        assert len(result) == len(data.index)
        tm.assert_frame_equal(result, data)

    return temporary


def test_load_data_sqlalchemy(
    alchemy_backend, alchemy_con, alchemy_temp_table
):
    sch = ibis.schema(
        [
            ('first_name', 'string'),
            ('last_name', 'string'),
            ('department_name', 'string'),
            ('salary', 'float64'),
        ]
    )

    df = pd.DataFrame(
        {
            'first_name': ['A', 'B', 'C'],
            'last_name': ['D', 'E', 'F'],
            'department_name': ['AA', 'BB', 'CC'],
            'salary': [100.0, 200.0, 300.0],
        }
    )
    alchemy_con.create_table(alchemy_temp_table, schema=sch)
    alchemy_con.load_data(alchemy_temp_table, df, if_exists='append')
    result = alchemy_con.table(alchemy_temp_table).execute()

    alchemy_backend.assert_frame_equal(df, result)


@mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t.string_col, [('string_col', dt.String)]),
        (
            lambda t: t[t.string_col, t.bigint_col],
            [('string_col', dt.String), ('bigint_col', dt.Int64)],
        ),
    ],
)
@mark.notimpl(["datafusion"])
def test_query_schema(ddl_backend, ddl_con, expr_fn, expected):
    expr = expr_fn(ddl_backend.functional_alltypes)

    # we might need a public API for it
    ast = ddl_con.compiler.to_ast(expr, ddl_backend.make_context())
    schema = ddl_con.ast_schema(ast)

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [
            (name, dtype(nullable=schema[name].nullable))
            for name, dtype in expected
        ]
    )
    assert schema.equals(expected)


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["sqlite"])
@pytest.mark.never(
    ["dask", "pandas"],
    reason="dask and pandas do not support SQL",
)
def test_sql(con):
    # execute the expression using SQL query
    expr = con.sql("SELECT * FROM functional_alltypes LIMIT 10")
    result = expr.execute()
    assert len(result) == 10


@mark.notimpl(["clickhouse", "datafusion"])
def test_create_table_from_schema(con, new_schema, temp_table):
    con.create_table(temp_table, schema=new_schema)

    t = con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@mark.notimpl(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
    ]
)
def test_rename_table(con, temp_table, new_schema):
    temp_table_original = f'{temp_table}_original'
    con.create_table(temp_table_original, schema=new_schema)
    try:
        t = con.table(temp_table_original)
        t.rename(temp_table)

        assert con.table(temp_table) is not None
        assert temp_table in con.list_tables()
    finally:
        con.drop_table(temp_table_original, force=True)
        con.drop_table(temp_table, force=True)


@mark.notimpl(["clickhouse", "datafusion"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
def test_nullable_input_output(con, temp_table):
    sch = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', ibis.expr.datatypes.int64(nullable=False)),
            ('baz', 'boolean'),
        ]
    )

    con.create_table(temp_table, schema=sch)

    t = con.table(temp_table)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


@mark.notimpl(
    ["clickhouse", "datafusion", "duckdb", "mysql", "postgres", "sqlite"]
)
@mark.notyet(["pyspark"])
def test_create_drop_view(ddl_con, temp_view):
    # setup
    table_name = 'functional_alltypes'
    expr = ddl_con.table(table_name).limit(1)

    # create a new view
    ddl_con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in ddl_con.list_tables()

    t_expr = ddl_con.table(table_name)
    v_expr = ddl_con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)


@mark.notimpl(["postgres", "mysql", "clickhouse", "datafusion"])
def test_separate_database(
    ddl_con, alternate_current_database, current_data_db
):
    # using alternate_current_database switches "con" current
    #  database to a temporary one until a test is over
    tmp_db = ddl_con.database(alternate_current_database)
    # verifying we can open another db which isn't equal to current
    db = ddl_con.database(current_data_db)
    assert db.name == current_data_db
    assert tmp_db.name == alternate_current_database


@pytest.fixture
def employee_empty_temp_table(alchemy_con, test_employee_schema):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


@pytest.fixture
def employee_data_1_temp_table(
    alchemy_con,
    test_employee_schema,
    test_employee_data_1,
):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
        data=test_employee_data_1,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


@pytest.fixture
def employee_data_2_temp_table(
    alchemy_con,
    test_employee_schema,
    test_employee_data_2,
):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


def test_insert_no_overwrite_from_dataframe(
    alchemy_con,
    test_employee_data_2,
    employee_empty_temp_table,
):
    temporary = alchemy_con.table(employee_empty_temp_table)
    alchemy_con.insert(
        employee_empty_temp_table,
        obj=test_employee_data_2,
        overwrite=False,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, test_employee_data_2)


def test_insert_overwrite_from_dataframe(
    alchemy_con,
    employee_data_1_temp_table,
    test_employee_data_2,
):
    temporary = alchemy_con.table(employee_data_1_temp_table)

    alchemy_con.insert(
        employee_data_1_temp_table,
        obj=test_employee_data_2,
        overwrite=True,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, test_employee_data_2)


def test_insert_no_overwite_from_expr(
    alchemy_con,
    employee_empty_temp_table,
    employee_data_2_temp_table,
):
    temporary = alchemy_con.table(employee_empty_temp_table)
    from_table = alchemy_con.table(employee_data_2_temp_table)

    alchemy_con.insert(
        employee_empty_temp_table,
        obj=from_table,
        overwrite=False,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, from_table.execute())


def test_insert_overwrite_from_expr(
    alchemy_con,
    employee_data_1_temp_table,
    employee_data_2_temp_table,
):
    temporary = alchemy_con.table(employee_data_1_temp_table)
    from_table = alchemy_con.table(employee_data_2_temp_table)

    alchemy_con.insert(
        employee_data_1_temp_table,
        obj=from_table,
        overwrite=True,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, from_table.execute())


def test_list_databases(alchemy_con):
    # Every backend has its own databases
    TEST_DATABASES = {
        'sqlite': ['main'],
        'postgres': ['postgres', 'ibis_testing'],
        'mysql': ['ibis_testing', 'information_schema'],
        'duckdb': ['information_schema', 'main', 'temp'],
    }
    assert alchemy_con.list_databases() == TEST_DATABASES[alchemy_con.name]


def test_list_schemas(alchemy_con):
    with pytest.warns(FutureWarning):
        alchemy_con.list_schemas()


def test_verify(ddl_backend, ddl_con):
    expr = ddl_con.table('functional_alltypes').double_col.sum()

    with pytest.warns(FutureWarning):
        assert expr.verify()

    with pytest.warns(FutureWarning):
        assert ddl_backend.api.verify(expr)


@pytest.mark.never(["duckdb"], reason="duckdb supports approximate median")
def test_not_verify(alchemy_con, alchemy_backend):
    # There is no expression that can't be compiled to any backend
    # Testing `not verify()` only for an expression not supported in postgres
    expr = alchemy_con.table('functional_alltypes').double_col.approx_median()
    with pytest.warns(FutureWarning):
        assert not expr.verify()

    with pytest.warns(FutureWarning):
        assert not alchemy_backend.api.verify(expr)


@pytest.mark.never(
    ["postgres", "mysql"],
    reason="postgres and mysql do not support in-memory tables",
    raises=sa.exc.OperationalError,
)
def test_in_memory(alchemy_backend):
    con = getattr(ibis, alchemy_backend.name()).connect(":memory:")
    table_name = f"t{guid()[:6]}"
    con.raw_sql(f"CREATE TABLE {table_name} (x int)")
    try:
        assert table_name in con.list_tables()
    finally:
        con.raw_sql(f"DROP TABLE IF EXISTS {table_name}")
        assert table_name not in con.list_tables()


@pytest.mark.parametrize(
    "coltype", [dt.uint8, dt.uint16, dt.uint32, dt.uint64]
)
@pytest.mark.notyet(
    ["postgres", "mysql", "sqlite"],
    raises=TypeError,
    reason="postgres, mysql and sqlite do not support unsigned integer types",
)
def test_unsigned_integer_type(alchemy_con, coltype):
    tname = guid()
    alchemy_con.create_table(
        tname, schema=ibis.schema(dict(a=coltype)), force=True
    )
    try:
        assert tname in alchemy_con.list_tables()
    finally:
        alchemy_con.drop_table(tname, force=True)


@pytest.mark.backend
@pytest.mark.parametrize(
    "url",
    [
        param(
            "clickhouse://default@localhost:9000/ibis_testing",
            marks=mark.clickhouse,
            id="clickhouse",
        ),
        param(
            "dask://",
            marks=[mark.dask, mark.xfail(raises=NotImplementedError)],
            id="dask",
        ),
        param(
            "datafusion://",
            marks=[mark.datafusion, mark.xfail(raises=NotImplementedError)],
            id="datafusion",
        ),
        param(
            "impala://localhost:21050/ibis_testing",
            marks=mark.impala,
            id="impala",
        ),
        param(
            "mysql://ibis:ibis@localhost:3306/ibis_testing",
            marks=mark.mysql,
            id="mysql",
        ),
        param(
            "pandas://",
            marks=[mark.pandas, mark.xfail(raises=NotImplementedError)],
            id="pandas",
        ),
        param(
            "postgres://postgres:postgres@localhost:5432/ibis_testing",
            marks=mark.postgres,
            id="postgres",
        ),
        param(
            "pyspark://?spark.app.name=test-pyspark",
            marks=mark.pyspark,
            id="pyspark",
        ),
        param(
            "pyspark://my-warehouse-dir?spark.app.name=test-pyspark",
            marks=mark.pyspark,
            id="pyspark_with_warehouse",
        ),
        param(
            "pyspark://my-warehouse-dir",
            marks=mark.pyspark,
            id="pyspark_with_warehouse_no_params",
        ),
    ],
)
def test_connect_url(url):
    con = ibis.connect(url)
    one = ibis.literal(1)
    assert con.execute(one) == 1


not_windows = pytest.mark.skipif(
    condition=platform.system() == "Windows",
    reason=(
        "windows prevents two connections to the same duckdb file even in "
        "the same process"
    ),
)


@pytest.mark.backend
@pytest.mark.parametrize(
    "url",
    [
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb",
            marks=[mark.duckdb, not_windows],
            id="duckdb_path",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=0",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_write_int",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=1",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_only_int",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=False",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_write_upper",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=True",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_only_upper",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=false",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_write_lower",
        ),
        param(
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb?read_only=true",
            marks=[mark.duckdb, not_windows],
            id="duckdb_read_only_lower",
        ),
        param("duckdb://", marks=mark.duckdb, id="duckdb_memory"),
        param(
            "duckdb://:memory:",
            marks=mark.duckdb,
            id="duckdb_memory_explicit",
        ),
        param(
            "duckdb://?read_only=1",
            marks=[
                mark.duckdb,
                pytest.mark.xfail(
                    reason="Read-only in-memory databases aren't supported",
                    raises=sa.exc.DBAPIError,
                ),
            ],
            id="duckdb_memory_read_only",
        ),
        param(
            "sqlite://ci/ibis-testing-data/ibis_testing.db",
            marks=mark.sqlite,
            id="sqlite_path",
        ),
        param("sqlite://", marks=mark.sqlite, id="sqlite_memory"),
        param(
            "sqlite://:memory:",
            marks=mark.sqlite,
            id="sqlite_memory_explicit",
        ),
    ],
)
def test_connect_file_url(url):
    con = ibis.connect(url)
    one = ibis.literal(1)
    assert con.execute(one) == 1


@not_windows
def test_invalid_connect():
    pytest.importorskip("duckdb")
    url = "?".join(
        [
            "duckdb://ci/ibis-testing-data/ibis_testing.ddb",
            "read_only=invalid_value",
        ]
    )
    with pytest.raises(ValueError):
        ibis.connect(url)


@pytest.mark.never(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
    ],
    reason="backend isn't file-based",
)
def test_deprecated_path_argument(backend, tmp_path):
    with pytest.warns(UserWarning, match="The `path` argument is deprecated"):
        getattr(ibis, backend.name()).connect(path=str(tmp_path / "test.db"))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(
            ibis.memtable([(1, 2.0, "3")], columns=list("abc")),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")),
            id="simple",
        ),
        param(
            ibis.memtable([(1, 2.0, "3")]),
            pd.DataFrame([(1, 2.0, "3")], columns=["col0", "col1", "col2"]),
            id="simple_auto_named",
        ),
        param(
            ibis.memtable(
                [(1, 2.0, "3")],
                schema=ibis.schema(dict(a="int8", b="float32", c="string")),
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="simple_schema",
        ),
        param(
            ibis.memtable(
                pd.DataFrame({"a": [1], "b": [2.0], "c": ["3"]}).astype(
                    {"a": "int8", "b": "float32"}
                )
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="dataframe",
        ),
        param(
            ibis.memtable([dict(a=1), dict(a=2)]),
            pd.DataFrame({"a": [1, 2]}),
            id="list_of_dicts",
        ),
    ],
)
@pytest.mark.notyet(
    ["mysql", "sqlite"],
    reason="SQLAlchemy generates incorrect code for `VALUES` projections.",
    raises=(sa.exc.ProgrammingError, sa.exc.OperationalError),
)
@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_in_memory_table(backend, con, expr, expected):
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["mysql", "sqlite"],
    reason="SQLAlchemy generates incorrect code for `VALUES` projections.",
    raises=(sa.exc.ProgrammingError, sa.exc.OperationalError),
)
@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_filter_memory_table(backend, con):
    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.filter(t.x > 1)
    expected = pd.DataFrame({"x": [3, 5], "y": [4, 6]})
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["mysql", "sqlite"],
    reason="SQLAlchemy generates incorrect code for `VALUES` projections.",
    raises=(sa.exc.ProgrammingError, sa.exc.OperationalError),
)
@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_agg_memory_table(con):
    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.x.count()
    result = con.execute(expr)
    assert result == 3


@pytest.mark.parametrize(
    "t",
    [
        param(
            ibis.memtable([("a", 1.0)], columns=["a", "b"]),
            id="python",
        ),
        param(
            ibis.memtable(pd.DataFrame([("a", 1.0)], columns=["a", "b"])),
            id="pandas",
        ),
    ],
)
@pytest.mark.notimpl(["clickhouse", "dask", "datafusion", "pandas"])
def test_create_from_in_memory_table(con, t):
    tmp_name = guid()
    con.create_table(tmp_name, t)
    try:
        assert tmp_name in con.list_tables()
    finally:
        con.drop_table(tmp_name)
        assert tmp_name not in con.list_tables()


def test_default_backend_no_duckdb(backend):
    # backend is used to ensure that this test runs in CI in the setting
    # where only the dependencies for a a given backend are installed

    # if duckdb is available then this test won't fail and so we skip it
    try:
        import duckdb  # noqa: F401

        pytest.skip(
            "duckdb is installed; it will be used as the default backend"
        )
    except ImportError:
        pass

    df = pd.DataFrame({'a': [1, 2, 3]})
    t = ibis.memtable(df)
    expr = t.a.sum()

    # run this twice to ensure that we hit the optimizations in
    # `_default_backend`
    for _ in range(2):
        with pytest.raises(
            com.IbisError,
            match="Expression depends on no backends",
        ):
            expr.execute()


@pytest.mark.duckdb
def test_default_backend():
    pytest.importorskip("duckdb")

    df = pd.DataFrame({'a': [1, 2, 3]})
    t = ibis.memtable(df)
    expr = t.a.sum()
    # run this twice to ensure that we hit the optimizations in
    # `_default_backend`
    for _ in range(2):
        assert expr.execute() == df.a.sum()

    sql = ibis.to_sql(expr)
    rx = """\
SELECT
  SUM\\((\\w+)\\.a\\) AS sum
FROM \\w+ AS \\1"""
    assert re.match(rx, sql) is not None
