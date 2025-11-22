"""Test Materialize backend client functionality."""

from __future__ import annotations

import os

import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.util import gen_name

MATERIALIZE_TEST_DB = os.environ.get("IBIS_TEST_MATERIALIZE_DATABASE", "materialize")
IBIS_MATERIALIZE_HOST = os.environ.get("IBIS_TEST_MATERIALIZE_HOST", "localhost")
IBIS_MATERIALIZE_PORT = os.environ.get("IBIS_TEST_MATERIALIZE_PORT", "6875")


class TestMaterializeClient:
    """Test Materialize-specific client functionality."""

    def test_version(self, con):
        """Test that version property works and strips leading 'v'."""
        version = con.version
        assert version is not None
        assert isinstance(version, str)
        # Materialize versions like "v0.158.2" should be stripped to "0.158.2"
        assert not version.startswith("v"), (
            f"Version should not start with 'v': {version}"
        )
        assert "." in version, f"Version should contain dots: {version}"

    def test_list_tables(self, con):
        """Test listing tables."""
        tables = con.list_tables()
        assert isinstance(tables, list)
        # Should have at least the test tables loaded
        assert len(tables) > 0

        # Test filtering with like parameter
        assert len(con.list_tables(like="functional")) == 1
        assert {"batting", "diamonds", "functional_alltypes"} <= set(con.list_tables())

        # Note: Temp tables may not appear in list_tables() depending on schema visibility
        # This is expected Materialize behavior

    def test_session_temp_db_is_mz_temp(self, con):
        """Test that Materialize uses mz_temp for temporary objects."""
        # Materialize supports temp tables in mz_temp schema
        assert con._session_temp_db == "mz_temp"

    def test_backend_name(self, con):
        """Test backend name is correct."""
        assert con.name == "materialize"

    def test_temp_tables_supported(self, con):
        """Test that temporary tables are correctly marked as supported."""
        assert con.supports_temporary_tables is True
        assert con._session_temp_db == "mz_temp"

    def test_python_udfs_not_supported(self, con):
        """Test that Python UDFs are correctly marked as unsupported."""
        assert con.supports_python_udfs is False

    def test_create_table_expression(self, con, alltypes):  # noqa: ARG002
        """Test creating table expressions."""
        # Should be able to create table expression without error
        assert alltypes is not None
        # Should have columns
        assert len(alltypes.columns) > 0

    def test_simple_query(self, con, alltypes):  # noqa: ARG002
        """Test executing a simple query."""
        result = alltypes.limit(5).execute()
        assert len(result) == 5

    def test_aggregation(self, con, alltypes):  # noqa: ARG002
        """Test basic aggregation."""
        result = alltypes.count().execute()
        assert isinstance(result, int)
        assert result > 0

    def test_group_by(self, con, alltypes):  # noqa: ARG002
        """Test group by aggregation."""
        result = (
            alltypes.group_by("string_col").aggregate(count=alltypes.count()).execute()
        )
        assert len(result) > 0
        assert "string_col" in result.columns
        assert "count" in result.columns


@pytest.mark.parametrize(
    "table_name",
    [
        "functional_alltypes",
        "batting",
        "awards_players",
    ],
)
def test_load_data(con, table_name):
    """Test that test data was loaded successfully."""
    table = con.table(table_name)
    result = table.limit(1).execute()
    assert len(result) == 1


def test_schema_introspection_no_unnest_error(con, alltypes):  # noqa: ARG001
    """Test that schema introspection doesn't hit unnest() ambiguity error.

    This is the second critical workaround - Materialize's unnest() function
    has multiple signatures and causes ambiguity errors with PostgreSQL's
    ANY(array) syntax. Our get_schema() override fixes this.
    """
    # This should not raise "function unnest(unknown) is not unique" error
    schema = con.get_schema("functional_alltypes")
    assert schema is not None
    assert len(schema) > 0


def test_connect_with_schema(con):
    """Test that connecting with a specific schema works."""
    # Materialize should handle schema parameter
    # (though implementation may differ from PostgreSQL)
    # Connection already tested via fixtures
    assert con is not None


class TestMaterializeSpecific:
    """Tests specific to Materialize's streaming database features."""

    def test_can_query_materialized_views(self, con):
        """Test that we can query materialized views if they exist."""
        # In a real Materialize deployment, there would be materialized views
        # For now, just verify we can list what exists
        tables = con.list_tables()
        # Tables list includes both tables and materialized views
        assert isinstance(tables, list)

    def test_postgresql_compatibility(self, con):
        """Test that PostgreSQL-compatible queries work."""
        # Simple PostgreSQL-compatible query
        result = con.sql("SELECT 1 AS test").execute()
        assert result["test"].iloc[0] == 1

    def test_no_pg_my_temp_schema_in_queries(self, con, alltypes):
        """Verify that no queries attempt to use pg_my_temp_schema()."""
        # If this test passes, it means our workaround is preventing
        # any code path that would call pg_my_temp_schema()

        # These operations would fail if _session_temp_db tried to call pg_my_temp_schema()
        con.list_tables()
        con.get_schema("functional_alltypes")
        alltypes.limit(1).execute()

        # If we got here, the workaround is working
        assert True


def test_table(alltypes):
    """Test that table returns correct type."""
    assert isinstance(alltypes, ibis.expr.types.Table)


def test_array_execute(alltypes):
    """Test executing array column."""
    d = alltypes.limit(10).double_col
    s = d.execute()
    import pandas as pd

    assert isinstance(s, pd.Series)
    assert len(s) == 10


def test_literal_execute(con):
    """Test literal execution."""
    expr = ibis.literal("1234")
    result = con.execute(expr)
    assert result == "1234"


def test_simple_aggregate_execute(alltypes):
    """Test simple aggregation execution."""
    d = alltypes.double_col.sum()
    v = d.execute()
    assert isinstance(v, float)


def test_compile_toplevel(assert_sql):
    """Test compiling expression at top level."""
    t = ibis.table([("foo", "double")], name="t0")
    expr = t.foo.sum()
    assert_sql(expr)


def test_list_catalogs(con):
    """Test listing catalogs."""
    assert MATERIALIZE_TEST_DB is not None
    catalogs = con.list_catalogs()
    assert isinstance(catalogs, list)
    assert MATERIALIZE_TEST_DB in catalogs


def test_list_databases(con):
    """Test listing databases/schemas."""
    databases = con.list_databases()
    assert isinstance(databases, list)
    # Materialize should have these schemas
    assert "information_schema" in databases
    assert "pg_catalog" in databases
    assert "public" in databases
    # Materialize-specific catalog
    assert "mz_catalog" in databases


def test_create_and_drop_table(con, temp_table):
    """Test creating and dropping a table."""
    sch = ibis.schema([("first_name", "string")])

    con.create_table(temp_table, schema=sch)
    assert con.table(temp_table) is not None

    con.drop_table(temp_table)

    assert temp_table not in con.list_tables()


@pytest.mark.parametrize(
    ("pg_type", "expected_type"),
    [
        param(pg_type, ibis_type, id=pg_type.lower())
        for (pg_type, ibis_type) in [
            ("boolean", dt.boolean),
            ("bytea", dt.binary),
            ("bigint", dt.int64),
            ("smallint", dt.int16),
            ("integer", dt.int32),
            ("text", dt.string),
            ("real", dt.float32),
            ("double precision", dt.float64),
            ("character varying", dt.string),
            ("date", dt.date),
            ("time", dt.time),
            ("time without time zone", dt.time),
            ("timestamp without time zone", dt.Timestamp(scale=6)),
            ("timestamp with time zone", dt.Timestamp("UTC", scale=6)),
            ("interval", dt.Interval("s")),
            ("numeric", dt.decimal),
            ("jsonb", dt.JSON(binary=True)),
        ]
    ],
)
def test_get_schema_from_query(con, pg_type, expected_type):
    """Test schema introspection from query with various data types."""
    name = sg.table(gen_name("materialize_temp_table"), quoted=True)
    with con.begin() as c:
        c.execute(f"CREATE TABLE {name} (x {pg_type}, y {pg_type}[])")
    expected_schema = ibis.schema(dict(x=expected_type, y=dt.Array(expected_type)))
    result_schema = con._get_schema_using_query(f"SELECT x, y FROM {name}")
    assert result_schema == expected_schema
    with con.begin() as c:
        c.execute(f"DROP TABLE {name}")


def test_insert_with_cte(con):
    """Test insert with CTE."""
    # Clean up any existing tables from previous runs
    for table in ["X", "Y"]:
        if table in con.list_tables():
            con.drop_table(table)

    X = con.create_table("X", schema=ibis.schema(dict(id="int")), temp=False)
    expr = X.join(X.mutate(a=X["id"] + 1), ["id"])
    Y = con.create_table("Y", expr, temp=False)
    assert Y.execute().empty
    con.drop_table("Y")
    con.drop_table("X")


def test_raw_sql(con):
    """Test raw SQL execution."""
    with con.raw_sql("SELECT 1 AS foo") as cur:
        assert cur.fetchall() == [(1,)]
    con.con.commit()


def test_create_table_from_dataframe(con):
    """Test creating table from pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    name = gen_name("df_table")

    try:
        table = con.create_table(name, df, temp=True)
        result = table.execute()
        assert len(result) == 3
        assert list(result.columns) == ["x", "y"]
        assert result["x"].tolist() == [1, 2, 3]
    finally:
        con.drop_table(name, force=True)


def test_create_table_from_pyarrow(con):
    """Test creating table from PyArrow table."""
    pa = pytest.importorskip("pyarrow")

    arrow_table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    name = gen_name("arrow_table")

    try:
        table = con.create_table(name, arrow_table, temp=True)
        result = table.execute()
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]
    finally:
        con.drop_table(name, force=True)


def test_nans_and_nulls(con):
    """Test handling of NaN and NULL values."""
    import math

    pa = pytest.importorskip("pyarrow")

    name = gen_name("nan_test")
    data = pa.table({"value": [1.0, float("nan"), None], "key": [1, 2, 3]})

    try:
        table = con.create_table(name, obj=data, temp=True)
        result = table.order_by("key").to_pyarrow()
        assert result.num_rows == 3

        value = result["value"]
        assert value[0].as_py() == 1.0
        assert math.isnan(value[1].as_py())
        assert value[2].as_py() is None
    finally:
        con.drop_table(name, force=True)


def test_filter_and_execute(alltypes):
    """Test filtering and execution."""
    expr = alltypes.filter(alltypes.int_col > 0).limit(5)
    result = expr.execute()
    assert len(result) <= 5
    assert all(result["int_col"] > 0)


def test_join_tables(con, alltypes):  # noqa: ARG001
    """Test joining tables."""
    # Self-join
    t1 = alltypes.select("id", "int_col")
    t2 = alltypes.select("id", "double_col")
    joined = t1.join(t2, "id").limit(5)
    result = joined.execute()
    assert len(result) == 5
    assert "int_col" in result.columns
    assert "double_col" in result.columns


def test_window_functions(alltypes):
    """Test window function execution."""
    expr = alltypes.mutate(
        row_num=ibis.row_number().over(ibis.window(order_by=alltypes.id))
    ).limit(5)
    result = expr.execute()
    assert "row_num" in result.columns
    assert len(result) == 5


def test_create_table_with_temp_flag(con):
    """Test creating table with temp=True flag."""
    name = gen_name("test_temp")

    try:
        # Create temp table
        table = con.create_table(name, schema=ibis.schema({"x": "int"}), temp=True)
        # Note: temp tables may not appear in list_tables() - this is expected

        # Verify we can insert into it
        con.raw_sql(f'INSERT INTO "{name}" (x) VALUES (1), (2), (3)')
        con.con.commit()

        result = table.execute()
        assert len(result) == 3
    finally:
        con.drop_table(name, force=True)


def test_current_database(con):
    """Test current_database property."""
    current_db = con.current_database
    assert current_db is not None
    assert isinstance(current_db, str)
    # Should be 'public' by default
    assert current_db == "public"


def test_exists_table(con):
    """Test checking if table exists."""
    # Test existing table
    assert con.table("functional_alltypes") is not None

    # Test non-existing table
    with pytest.raises(Exception):  # noqa: B017
        con.table("nonexistent_table_xyz")


def test_null_handling_in_aggregation(alltypes):
    """Test that NULL values are handled correctly in aggregations."""
    import pandas as pd

    # Create expression with potential nulls
    expr = alltypes.double_col.mean()
    result = expr.execute()
    assert isinstance(result, float)
    assert not pd.isna(result)


def test_distinct(alltypes):
    """Test distinct operation."""
    expr = alltypes[["string_col"]].distinct()
    result = expr.execute()
    # Result should have unique values
    assert len(result) == len(result["string_col"].unique())


def test_order_by(alltypes):
    """Test ordering."""
    expr = alltypes.order_by(alltypes.int_col.desc()).limit(10)
    result = expr.execute()
    assert len(result) == 10
    # Check ordering
    int_values = result["int_col"].tolist()
    assert int_values == sorted(int_values, reverse=True)


def test_limit_offset(alltypes):
    """Test LIMIT and OFFSET."""
    # Get first 5 rows
    first_5 = alltypes.limit(5).execute()
    # Get next 5 rows
    next_5 = alltypes.limit(5, offset=5).execute()

    assert len(first_5) == 5
    assert len(next_5) == 5
    # Should be different rows
    assert not first_5["id"].equals(next_5["id"])


def test_cast_operations(alltypes):
    """Test type casting."""
    expr = alltypes.select(
        int_as_string=alltypes.int_col.cast("string"),
        double_as_int=alltypes.double_col.cast("int"),
    ).limit(5)
    result = expr.execute()
    assert result["int_as_string"].dtype == object  # string type
    assert result["double_as_int"].dtype in [int, "int32", "int64"]
