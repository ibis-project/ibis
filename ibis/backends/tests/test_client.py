import os
import platform
import re
import string
import subprocess
import sys

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import rich.console
import sqlalchemy as sa
from pytest import mark, param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.util import guid


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


def _create_temp_table_with_schema(con, temp_table_name, schema, data=None):
    temporary = con.create_table(temp_table_name, schema=schema)
    assert temporary.to_pandas().empty

    if data is not None and isinstance(data, pd.DataFrame):
        assert not data.empty
        tmp = con.create_table(temp_table_name, data, overwrite=True)
        result = tmp.to_pandas()
        assert len(result) == len(data.index)
        tm.assert_frame_equal(
            result.sort_values(result.columns[0]).reset_index(drop=True),
            data.sort_values(result.columns[0]).reset_index(drop=True),
        )
        return tmp

    return temporary


def test_load_data_sqlalchemy(alchemy_backend, alchemy_con, alchemy_temp_table):
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
    alchemy_con.create_table(alchemy_temp_table, df, schema=sch, overwrite=True)
    result = (
        alchemy_con.table(alchemy_temp_table)
        .execute()
        .sort_values("first_name")
        .reset_index(drop=True)
    )

    alchemy_backend.assert_frame_equal(df, result)


@mark.parametrize(
    ('expr_fn', 'expected'),
    [
        param(lambda t: t.string_col, [('string_col', dt.String)], id="column"),
        param(
            lambda t: t[t.string_col, t.bigint_col],
            [('string_col', dt.String), ('bigint_col', dt.Int64)],
            id="table",
        ),
    ],
)
def test_query_schema(ddl_backend, expr_fn, expected):
    expr = expr_fn(ddl_backend.functional_alltypes)

    # we might need a public API for it
    schema = expr.as_table().schema()

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [(name, dtype(nullable=schema[name].nullable)) for name, dtype in expected]
    )
    assert schema.equals(expected)


@pytest.mark.notimpl(["datafusion", "snowflake", "polars", "mssql"])
@pytest.mark.notyet(["sqlite"])
@pytest.mark.never(
    ["dask", "pandas"],
    reason="dask and pandas do not support SQL",
)
def test_sql(backend, con):
    # execute the expression using SQL query
    table = backend.format_table("functional_alltypes")
    expr = con.sql(f"SELECT * FROM {table} LIMIT 10")
    result = expr.execute()
    assert len(result) == 10


backend_type_mapping = {
    "bigquery": {
        # backend only implements int64
        dt.int32: dt.int64
    }
}


@mark.notimpl(["clickhouse", "datafusion", "polars", "druid"])
def test_create_table_from_schema(con, new_schema, temp_table):
    new_table = con.create_table(temp_table, schema=new_schema)
    backend_mapping = backend_type_mapping.get(con.name, dict())

    for column_name, column_type in new_table.schema().items():
        assert (
            backend_mapping.get(new_schema[column_name], new_schema[column_name])
            == column_type
        )


@pytest.fixture(scope="session")
def tmpcon(alchemy_con):
    """A fixture to scope the connection for temp table testing.

    This prevents resetting the connection for subsequent tests that may depend
    on connection state persisting across tests.
    """
    return alchemy_con._from_url(alchemy_con.con.url)


@mark.notimpl(["trino", "druid"])
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
def test_create_temporary_table_from_schema(tmpcon, new_schema):
    temp_table = f"_{guid()}"
    table = tmpcon.create_table(temp_table, schema=new_schema, temp=True)

    # verify table exist in the current session
    backend_mapping = backend_type_mapping.get(tmpcon.name, dict())
    for column_name, column_type in table.schema().items():
        assert (
            backend_mapping.get(new_schema[column_name], new_schema[column_name])
            == column_type
        )

    tmpcon.reconnect()
    # verify table no longer exist after reconnect
    assert temp_table not in tmpcon.tables.keys()


@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
    ]
)
@mark.broken(["druid"], reason="sqlalchemy dialect is broken")
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


@mark.notimpl(["bigquery", "clickhouse", "datafusion", "polars", "druid"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
@mark.notyet(
    ["trino"], reason="trino doesn't support NOT NULL in its in-memory catalog"
)
def test_nullable_input_output(con, temp_table):
    sch = ibis.schema(
        [('foo', 'int64'), ('bar', dt.int64(nullable=False)), ('baz', 'boolean')]
    )
    t = con.create_table(temp_table, schema=sch)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


@mark.notimpl(["datafusion", "druid", "polars"])
def test_create_drop_view(ddl_con, temp_view):
    # setup
    table_name = 'functional_alltypes'
    try:
        expr = ddl_con.table(table_name)
    except KeyError:
        table_name = table_name.upper()
        expr = ddl_con.table(table_name)

    expr = expr.limit(1)

    # create a new view
    ddl_con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in ddl_con.list_tables()

    t_expr = ddl_con.table(table_name)
    v_expr = ddl_con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)


@mark.notimpl(["postgres", "mysql", "datafusion", "polars"])
def test_separate_database(ddl_con, alternate_current_database, current_data_db):
    # using alternate_current_database switches "con" current
    #  database to a temporary one until a test is over
    tmp_db = ddl_con.database(alternate_current_database)
    # verifying we can open another db which isn't equal to current
    db = ddl_con.database(current_data_db)
    assert db.name == current_data_db
    assert tmp_db.name == alternate_current_database


@pytest.fixture
def employee_empty_temp_table(alchemy_con, test_employee_schema):
    temp_table_name = f"temp_to_table_{guid()[:6]}"
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
    temp_table_name = f"temp_to_table_{guid()[:6]}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
        data=test_employee_data_1,
    )
    assert temp_table_name in alchemy_con.list_tables()
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
    temp_table_name = f"temp_to_table_{guid()[:6]}"
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
    tm.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        test_employee_data_2.sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notyet(
    ["trino"],
    reason="Connector doesn't support deletion (required for overwrite=True)",
    raises=sa.exc.ProgrammingError,
)
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
    tm.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        test_employee_data_2.sort_values("first_name").reset_index(drop=True),
    )


def test_insert_no_overwrite_from_expr(
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
    tm.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        from_table.execute().sort_values("first_name").reset_index(drop=True),
    )


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
    tm.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        from_table.execute().sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notyet(
    ["trino"], reason="memory connector doesn't allow writing to tables"
)
def test_insert_overwrite_from_list(
    alchemy_con,
    employee_data_1_temp_table,
):
    def _emp(a, b, c, d):
        return dict(first_name=a, last_name=b, department_name=c, salary=d)

    alchemy_con.insert(
        employee_data_1_temp_table,
        [
            _emp('Adam', 'Smith', 'Accounting', 50000.0),
            _emp('Mohammed', 'Ali', 'Boxing', 150000),
            _emp('María', 'Gonzalez', 'Engineering', 100000.0),
        ],
        overwrite=True,
    )

    assert len(alchemy_con.table(employee_data_1_temp_table).execute()) == 3


@pytest.mark.broken(
    ["snowflake"],
    reason="quoting is incorrect because pandas doesn't expose sqlalchemy column quoting",
)
def test_insert_from_memtable(alchemy_con):
    df = pd.DataFrame({"x": range(3)})
    table_name = "memtable_test"
    mt = ibis.memtable(df)
    alchemy_con.create_table(table_name, schema=mt.schema())
    alchemy_con.insert(table_name, mt)
    alchemy_con.insert(table_name, mt)

    try:
        table = alchemy_con.tables[table_name]
        assert len(table.execute()) == 6
        assert alchemy_con.tables[table_name].schema() == ibis.schema({"x": "int64"})
    finally:
        with alchemy_con.begin() as con:
            con.exec_driver_sql(f"DROP TABLE IF EXISTS {table_name}")
        assert table_name not in alchemy_con.list_tables()


def test_list_databases(alchemy_con):
    # Every backend has its own databases
    test_databases = {
        "sqlite": {"main"},
        "postgres": {"postgres", "ibis_testing"},
        "mssql": {"INFORMATION_SCHEMA"},
        "mysql": {"ibis_testing", "information_schema"},
        "duckdb": {"information_schema", "main"},
        "snowflake": {"IBIS_TESTING"},
        "trino": {"default"},
    }
    assert test_databases[alchemy_con.name] <= set(alchemy_con.list_databases())


@pytest.mark.never(
    ["bigquery", "postgres", "mssql", "mysql", "snowflake"],
    reason="backend does not support client-side in-memory tables",
    raises=(sa.exc.OperationalError, TypeError),
)
@pytest.mark.notyet(
    ["trino"], reason="memory connector doesn't allow writing to tables"
)
def test_in_memory(alchemy_backend):
    con = getattr(ibis, alchemy_backend.name()).connect(":memory:")
    table_name = f"t{guid()[:6]}"
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TABLE {table_name} (x int)")
    try:
        assert table_name in con.list_tables()
    finally:
        with con.begin() as c:
            c.exec_driver_sql(f"DROP TABLE IF EXISTS {table_name}")
        assert table_name not in con.list_tables()


@pytest.mark.notyet(
    ["mssql", "mysql", "postgres", "snowflake", "sqlite", "trino"],
    raises=TypeError,
    reason="postgres, mysql and sqlite do not support unsigned integer types",
)
def test_unsigned_integer_type(alchemy_con):
    tname = f"t{guid()[:6]}"
    alchemy_con.create_table(
        tname,
        schema=ibis.schema(dict(a="uint8", b="uint16", c="uint32", d="uint64")),
        overwrite=True,
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
            "postgresql://postgres:postgres@localhost:5432/ibis_testing",
            marks=mark.postgres,
            id="postgresql",
        ),
        param(
            "pyspark://?spark.app.name=test-pyspark",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
            id="pyspark",
        ),
        param(
            "pyspark://my-warehouse-dir?spark.app.name=test-pyspark",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
            id="pyspark_with_warehouse",
        ),
        param(
            "pyspark://my-warehouse-dir",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
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


@pytest.fixture(params=["duckdb", "sqlite"])
def tmp_db(request, tmp_path):
    api = request.param
    mod = pytest.importorskip(api)
    db = tmp_path / "test.db"
    mod.connect(str(db)).execute("CREATE TABLE tmp_t AS SELECT 1 AS a").fetchall()
    return db


@pytest.mark.duckdb
@pytest.mark.parametrize(
    "url",
    [
        param(lambda p: p, id="no-scheme-duckdb-ext"),
        param(lambda p: f"duckdb://{p}", id="absolute-path"),
        param(
            lambda p: f"duckdb://{os.path.relpath(p)}",
            marks=[
                not_windows
            ],  # hard to test in CI since tmpdir & cwd are on different drives
            id="relative-path",
        ),
        param(lambda p: "duckdb://", id="in-memory-empty"),
        param(lambda p: "duckdb://:memory:", id="in-memory-explicit"),
        param(
            lambda p: f"duckdb://{p}?read_only=1",
            id="duckdb_read_write_int",
        ),
        param(
            lambda p: f"duckdb://{p}?read_only=False",
            id="duckdb_read_write_upper",
        ),
        param(
            lambda p: f"duckdb://{p}?read_only=false",
            id="duckdb_read_write_lower",
        ),
    ],
)
def test_connect_duckdb(url, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = os.path.abspath(tmp_path / "test.duckdb")
    with duckdb.connect(path):
        pass
    con = ibis.connect(url(path))
    one = ibis.literal(1)
    assert con.execute(one) == 1


@pytest.mark.sqlite
@pytest.mark.parametrize(
    "url, ext",
    [
        param(lambda p: p, "sqlite", id="no-scheme-sqlite-ext"),
        param(lambda p: p, "db", id="no-scheme-db-ext"),
        param(lambda p: f"sqlite://{p}", "db", id="absolute-path"),
        param(
            lambda p: f"sqlite://{os.path.relpath(p)}",
            "db",
            marks=[
                not_windows
            ],  # hard to test in CI since tmpdir & cwd are on different drives
            id="relative-path",
        ),
        param(lambda p: "sqlite://", "db", id="in-memory-empty"),
        param(lambda p: "sqlite://:memory:", "db", id="in-memory-explicit"),
    ],
)
def test_connect_sqlite(url, ext, tmp_path):
    import sqlite3

    path = os.path.abspath(tmp_path / f"test.{ext}")
    with sqlite3.connect(path):
        pass
    con = ibis.connect(url(path))
    one = ibis.literal(1)
    assert con.execute(one) == 1


@pytest.mark.duckdb
@pytest.mark.parametrize(
    "out_method, extension",
    [
        ("to_csv", "csv"),
        ("to_parquet", "parquet"),
    ],
)
def test_connect_local_file(out_method, extension, test_employee_data_1, tmp_path):
    getattr(test_employee_data_1, out_method)(tmp_path / f"out.{extension}")
    con = ibis.connect(tmp_path / f"out.{extension}")
    t = list(con.tables.values())[0]
    assert not t.head().execute().empty


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


@pytest.mark.parametrize("backend_name", ["sqlite", "duckdb"])
def test_deprecated_path_argument(backend_name, tmp_path):
    with pytest.warns(UserWarning, match="The `path` argument is deprecated"):
        getattr(ibis, backend_name).connect(path=str(tmp_path / "test.db"))


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
@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_in_memory_table(backend, con, expr, expected):
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_filter_memory_table(backend, con):
    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.filter(t.x > 1)
    expected = pd.DataFrame({"x": [3, 5], "y": [4, 6]})
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_agg_memory_table(con):
    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.x.count()
    result = con.execute(expr)
    assert result == 3


@pytest.mark.parametrize(
    "t",
    [
        param(ibis.memtable([("a", 1.0)], columns=["a", "b"]), id="python"),
        param(
            ibis.memtable(pd.DataFrame([("a", 1.0)], columns=["a", "b"])),
            id="pandas-memtable",
        ),
        param(pd.DataFrame([("a", 1.0)], columns=["a", "b"]), id="pandas"),
    ],
)
@pytest.mark.notimpl(["clickhouse", "dask", "datafusion", "pandas", "polars", "druid"])
def test_create_from_in_memory_table(backend, con, t):
    if backend.name() == "snowflake":
        pytest.skip("snowflake is unreliable here")
    tmp_name = f"t{guid()[:6]}"
    con.create_table(tmp_name, t)
    try:
        assert tmp_name in con.list_tables()
    finally:
        con.drop_table(tmp_name)
        assert tmp_name not in con.list_tables()


@pytest.mark.usefixtures("backend")
def test_default_backend_option():
    orig = ibis.options.default_backend
    ibis.options.default_backend = ibis.pandas
    try:
        backend = ibis.config._default_backend()
        assert backend.name == "pandas"
    finally:
        ibis.options.default_backend = orig


# backend is used to ensure that this test runs in CI in the setting
# where only the dependencies for a given backend are installed
@pytest.mark.usefixtures("backend")
def test_default_backend_no_duckdb():
    script = """\
import sys
sys.modules["duckdb"] = None

import ibis

t = ibis.memtable([{'a': 1}, {'a': 2}, {'a': 3}])
t.execute()"""

    args = [sys.executable, "-c", script]
    with pytest.raises(subprocess.CalledProcessError) as e:
        subprocess.check_output(args, stderr=subprocess.STDOUT, universal_newlines=True)
    assert (
        re.search(
            "You have used a function that relies on the default backend",
            e.value.output,
        )
        is not None
    )


@pytest.mark.usefixtures("backend")
def test_default_backend_no_duckdb_read_parquet():
    script = """\
import sys
sys.modules["duckdb"] = None

import ibis
ibis.read_parquet("foo.parquet")"""

    args = [sys.executable, "-c", script]
    with pytest.raises(subprocess.CalledProcessError) as e:
        subprocess.check_output(args, stderr=subprocess.STDOUT, universal_newlines=True)
    assert (
        re.search(
            "You have used a function that relies on the default backend",
            e.value.output,
        )
        is not None
    )


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

    sql = str(ibis.to_sql(expr))
    rx = """\
SELECT
  SUM\\((\\w+)\\.a\\) AS ".+"
FROM \\w+ AS \\1"""
    assert re.match(rx, sql) is not None


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_table(alltypes, dtype):
    expr = alltypes.group_by("string_col").int_col.sum().order_by("string_col")
    result = np.asarray(expr, dtype=dtype)
    expected = np.asarray(expr.execute(), dtype=dtype)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_column(alltypes, dtype):
    from ibis import _

    expr = alltypes.group_by("string_col").agg(int_col=_.int_col.sum()).int_col
    result = np.sort(np.asarray(expr, dtype=dtype))
    expected = np.sort(np.asarray(expr.execute(), dtype=dtype))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("interactive", [True, False])
def test_repr(alltypes, interactive, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", interactive)

    expr = alltypes.select("date_string_col")

    s = repr(expr)
    # no control characters
    assert all(c.isprintable() or c in "\n\r\t" for c in s)
    if interactive:
        assert "/" in s
    else:
        assert "/" not in s


@pytest.mark.parametrize("show_types", [True, False])
def test_interactive_repr_show_types(alltypes, show_types, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    monkeypatch.setattr(ibis.options.repr.interactive, "show_types", show_types)

    expr = alltypes.select("id")
    s = repr(expr)
    if show_types:
        assert "int" in s
    else:
        assert "int" not in s


@pytest.mark.parametrize("is_jupyter", [True, False])
def test_interactive_repr_max_columns(alltypes, is_jupyter, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)

    cols = {f"c_{i}": ibis._.id + i for i in range(50)}
    expr = alltypes.mutate(**cols).select(*cols)

    console = rich.console.Console(force_jupyter=is_jupyter)
    console.options.max_width = 80
    options = console.options.copy()

    # max_columns = 0
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # Default of 20 columns are written
        assert " c_19 " in text
    else:
        # width calculations truncates well before 20 columns
        assert " c_19 " not in text

    # max_columns = 3
    monkeypatch.setattr(ibis.options.repr.interactive, "max_columns", 3)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_2 " in text
    assert " c_3 " not in text

    # max_columns = None
    monkeypatch.setattr(ibis.options.repr.interactive, "max_columns", None)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # All columns written
        assert " c_49 " in text
    else:
        # width calculations still truncates
        assert " c_19 " not in text


@pytest.mark.parametrize("expr_type", ["table", "column"])
@pytest.mark.parametrize("interactive", [True, False])
def test_repr_mimebundle(alltypes, interactive, expr_type, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", interactive)

    if expr_type == "column":
        expr = alltypes.date_string_col
    else:
        expr = alltypes.select("date_string_col")

    reprs = expr._repr_mimebundle_(include=["text/plain", "text/html"], exclude=[])
    for format in ["text/plain", "text/html"]:
        if interactive:
            assert "r0.date_string_col" not in reprs[format]
        else:
            assert "r0.date_string_col" in reprs[format]


@pytest.mark.never(
    ["postgres", "mysql", "bigquery"],
    reason="These backends explicitly do support Geo operations",
)
def test_has_operation_no_geo(con):
    """Previously some backends mistakenly reported Geo operations as
    supported.

    Since most backends don't support Geo operations, we test that
    they're excluded here, skipping the few backends that explicitly do
    support them.
    """
    for op in [ops.GeoDistance, ops.GeoAsText, ops.GeoUnaryUnion]:
        assert not con.has_operation(op)


def test_get_backend(con, alltypes, monkeypatch):
    assert ibis.get_backend(alltypes) is con
    assert ibis.get_backend(alltypes.id.min()) is con

    with pytest.raises(com.IbisError, match="contains unbound tables"):
        ibis.get_backend(ibis.table({"x": "int"}))

    monkeypatch.setattr(ibis.options, "default_backend", con)
    assert ibis.get_backend() is con
    expr = ibis.literal(1) + 2
    assert ibis.get_backend(expr) is con


def test_set_backend(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(con)
    assert ibis.get_backend() is con


@pytest.mark.parametrize(
    "name",
    [
        param("duckdb", marks=mark.duckdb, id="duckdb"),
        param("polars", marks=mark.polars, id="polars"),
        param("sqlite", marks=mark.sqlite, id="sqlite"),
    ],
)
def test_set_backend_name(name, monkeypatch):
    # Don't need to test with all backends, only checking that things are
    # plumbed through correctly.
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(name)
    assert ibis.get_backend().name == name


@pytest.mark.parametrize(
    "url",
    [
        param(
            "clickhouse://default@localhost:9000/ibis_testing",
            marks=mark.clickhouse,
            id="clickhouse",
        ),
        param(
            "mysql://ibis:ibis@localhost:3306/ibis_testing",
            marks=mark.mysql,
            id="mysql",
        ),
        param(
            "postgres://postgres:postgres@localhost:5432/ibis_testing",
            marks=mark.postgres,
            id="postgres",
        ),
    ],
)
def test_set_backend_url(url, monkeypatch):
    # Don't need to test with all backends, only checking that things are
    # plumbed through correctly.
    monkeypatch.setattr(ibis.options, "default_backend", None)
    name = url.split("://")[0]
    ibis.set_backend(url)
    assert ibis.get_backend().name == name


@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "datafusion",
        "duckdb",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "sqlite",
    ],
    reason="backend doesn't support timestamp with scale parameter",
)
@pytest.mark.notimpl(["clickhouse"], reason="create table isn't implemented")
@pytest.mark.notimpl(
    ["snowflake"], reason="scale not implemented in ibis's snowflake backend"
)
@pytest.mark.broken(["druid"], reason="sqlalchemy dialect is broken")
def test_create_table_timestamp(con):
    schema = ibis.schema(
        dict(zip(string.ascii_letters, map("timestamp({:d})".format, range(10))))
    )
    name = f"timestamp_scale_{guid()}"
    con.create_table(name, schema=schema, overwrite=True)
    try:
        rows = con.raw_sql(f"DESCRIBE {name}").fetchall()
        result = ibis.schema((name, typ) for name, typ, *_ in rows)
        assert result == schema
    finally:
        con.drop_table(name, force=True)


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.never(
    ["pandas"],
    raises=AssertionError,
    reason="pandas doesn't cache anything and therefore does no ref counting",
)
def test_persist_expression_ref_count(con, alltypes):
    non_persisted_table = alltypes.mutate(test_column="calculation")
    persisted_table = non_persisted_table.cache()

    op = non_persisted_table.op()

    # ref count is unaffected without a context manager
    assert con._refs[op] == 1
    tm.assert_frame_equal(non_persisted_table.to_pandas(), persisted_table.to_pandas())
    assert con._refs[op] == 1


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
def test_persist_expression(alltypes):
    non_persisted_table = alltypes.mutate(test_column="calculation", other_calc="xyz")
    persisted_table = non_persisted_table.cache()
    tm.assert_frame_equal(non_persisted_table.to_pandas(), persisted_table.to_pandas())


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
def test_persist_expression_contextmanager(alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc"
    )
    with non_cached_table.cache() as cached_table:
        tm.assert_frame_equal(non_cached_table.to_pandas(), cached_table.to_pandas())


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.never(
    ["pandas"],
    raises=AssertionError,
    reason="pandas doesn't cache anything and therefore does no ref counting",
)
def test_persist_expression_contextmanager_ref_count(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    op = non_cached_table.op()
    with non_cached_table.cache() as cached_table:
        tm.assert_frame_equal(non_cached_table.to_pandas(), cached_table.to_pandas())
        assert con._refs[op] == 1
    assert con._refs[op] == 0


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.never(
    ["pandas"],
    raises=AssertionError,
    reason="pandas doesn't cache anything and therefore does no ref counting",
)
def test_persist_expression_multiple_refs(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    op = non_cached_table.op()
    with non_cached_table.cache() as cached_table:
        tm.assert_frame_equal(non_cached_table.to_pandas(), cached_table.to_pandas())
        with non_cached_table.cache() as nested_cached_table:
            assert not nested_cached_table.to_pandas().empty
            assert con._refs[op] == 2
        assert con._refs[op] == 1
    assert con._refs[op] == 0


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
def test_persist_expression_repeated_cache(alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    with non_cached_table.cache() as cached_table:
        with cached_table.cache() as nested_cached_table:
            assert not nested_cached_table.to_pandas().empty


@mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "bigquery",
        "impala",
        "pyspark",
        "trino",
        "druid",
    ]
)
@mark.notyet(
    ["sqlite"], reason="sqlite only support temporary tables in temporary databases"
)
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.never(["pandas"], reason="pandas does not need to release anything")
def test_persist_expression_release(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 3"
    )
    cached_table = non_cached_table.cache()
    cached_table.release()
    assert con._refs[non_cached_table.op()] == 0

    with pytest.raises(
        com.IbisError,
        match=r".+Did you call `\.release\(\)` twice on the same expression\?",
    ):
        cached_table.release()

    with pytest.raises(Exception, match=cached_table.op().name):
        cached_table.execute()
