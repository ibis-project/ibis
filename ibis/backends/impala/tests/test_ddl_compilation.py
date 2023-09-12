from __future__ import annotations

import pytest

import ibis
from ibis.backends.base.sql.ddl import (
    CTAS,
    CreateTableWithSchema,
    DropTable,
    InsertSelect,
)
from ibis.backends.impala import ddl
from ibis.backends.impala.compiler import ImpalaCompiler


@pytest.fixture
def t(mockcon):
    return mockcon.table("functional_alltypes")


def test_drop_table_compile(snapshot):
    statement = DropTable("foo", database="bar", must_exist=True)
    query = statement.compile()
    snapshot.assert_match(query, "out1.sql")

    statement = DropTable("foo", database="bar", must_exist=False)
    query = statement.compile()
    snapshot.assert_match(query, "out2.sql")


def test_select_basics(t, snapshot):
    name = "testing123456"

    expr = t.limit(10)
    select, _ = _get_select(expr)

    stmt = InsertSelect(name, select, database="foo")
    result = stmt.compile()
    snapshot.assert_match(result, "out1.sql")

    stmt = InsertSelect(name, select, database="foo", overwrite=True)
    result = stmt.compile()
    snapshot.assert_match(result, "out2.sql")


def test_load_data_unpartitioned(snapshot):
    path = "/path/to/data"
    stmt = ddl.LoadData("functional_alltypes", path, database="foo")

    result = stmt.compile()
    snapshot.assert_match(result, "out1.sql")

    stmt.overwrite = True
    result = stmt.compile()
    snapshot.assert_match(result, "out2.sql")


def test_load_data_partitioned(snapshot):
    path = "/path/to/data"
    part = {"year": 2007, "month": 7}
    part_schema = ibis.schema([("year", "int32"), ("month", "int32")])
    stmt = ddl.LoadData(
        "functional_alltypes",
        path,
        database="foo",
        partition=part,
        partition_schema=part_schema,
    )

    result = stmt.compile()
    snapshot.assert_match(result, "out1.sql")

    stmt.overwrite = True
    result = stmt.compile()
    snapshot.assert_match(result, "out2.sql")


def test_cache_table_pool_name(snapshot):
    statement = ddl.CacheTable("foo", database="bar")
    query = statement.compile()
    snapshot.assert_match(query, "out1.sql")

    statement = ddl.CacheTable("foo", database="bar", pool="my_pool")
    query = statement.compile()
    snapshot.assert_match(query, "out2.sql")


@pytest.fixture
def part_schema():
    return ibis.schema([("year", "int32"), ("month", "int32")])


@pytest.fixture
def table_name():
    return "tbl"


def test_add_partition(part_schema, table_name, snapshot):
    stmt = ddl.AddPartition(table_name, {"year": 2007, "month": 4}, part_schema)

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_add_partition_string_key(snapshot):
    part_schema = ibis.schema([("foo", "int32"), ("bar", "string")])
    stmt = ddl.AddPartition("tbl", {"foo": 5, "bar": "qux"}, part_schema)

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_drop_partition(part_schema, table_name, snapshot):
    stmt = ddl.DropPartition(table_name, {"year": 2007, "month": 4}, part_schema)

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_add_partition_with_props(part_schema, table_name, snapshot):
    props = {"location": "/users/foo/my-data"}
    stmt = ddl.AddPartition(
        table_name, {"year": 2007, "month": 4}, part_schema, **props
    )

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_alter_partition_properties(part_schema, table_name, snapshot):
    part = {"year": 2007, "month": 4}

    def _get_ddl_string(props):
        stmt = ddl.AlterPartition(table_name, part, part_schema, **props)
        return stmt.compile()

    result = _get_ddl_string({"location": "/users/foo/my-data"})
    snapshot.assert_match(result, "out1.sql")

    result = _get_ddl_string({"format": "avro"})
    snapshot.assert_match(result, "out2.sql")

    result = _get_ddl_string({"tbl_properties": {"bar": 2, "foo": "1"}})
    snapshot.assert_match(result, "out3.sql")

    result = _get_ddl_string({"serde_properties": {"baz": 3}})
    snapshot.assert_match(result, "out4.sql")


def test_alter_table_properties(part_schema, table_name, snapshot):
    part = {"year": 2007, "month": 4}

    def _get_ddl_string(props):
        stmt = ddl.AlterPartition(table_name, part, part_schema, **props)
        return stmt.compile()

    result = _get_ddl_string({"location": "/users/foo/my-data"})
    snapshot.assert_match(result, "out1.sql")

    result = _get_ddl_string({"format": "avro"})
    snapshot.assert_match(result, "out2.sql")

    result = _get_ddl_string({"tbl_properties": {"bar": 2, "foo": "1"}})
    snapshot.assert_match(result, "out3.sql")

    result = _get_ddl_string({"serde_properties": {"baz": 3}})
    snapshot.assert_match(result, "out4.sql")


@pytest.fixture
def expr(t):
    return t[t.bigint_col > 0]


def test_create_external_table_as(mockcon, snapshot):
    path = "/path/to/table"
    select, _ = _get_select(mockcon.table("test1"))
    statement = CTAS(
        "another_table",
        select,
        external=True,
        can_exist=False,
        path=path,
        database="foo",
    )
    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_with_location_compile(snapshot):
    path = "/path/to/table"
    schema = ibis.schema([("foo", "string"), ("bar", "int8"), ("baz", "int16")])
    statement = CreateTableWithSchema(
        "another_table",
        schema,
        can_exist=False,
        format="parquet",
        path=path,
        database="foo",
    )
    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_like_parquet(snapshot):
    directory = "/path/to/"
    path = "/path/to/parquetfile"
    statement = ddl.CreateTableParquet(
        "new_table",
        directory,
        example_file=path,
        can_exist=True,
        database="foo",
    )

    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_parquet_like_other(snapshot):
    # alternative to "LIKE PARQUET"
    directory = "/path/to/"
    example_table = "db.other"

    statement = ddl.CreateTableParquet(
        "new_table",
        directory,
        example_table=example_table,
        can_exist=True,
        database="foo",
    )

    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_parquet_with_schema(snapshot):
    directory = "/path/to/"

    schema = ibis.schema([("foo", "string"), ("bar", "int8"), ("baz", "int16")])

    statement = ddl.CreateTableParquet(
        "new_table",
        directory,
        schema=schema,
        external=True,
        can_exist=True,
        database="foo",
    )

    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_delimited(snapshot):
    path = "/path/to/files/"
    schema = ibis.schema(
        [
            ("a", "string"),
            ("b", "int32"),
            ("c", "double"),
            ("d", "decimal(12, 2)"),
        ]
    )

    stmt = ddl.CreateTableDelimited(
        "new_table",
        path,
        schema,
        delimiter="|",
        escapechar="\\",
        lineterminator="\0",
        database="foo",
        can_exist=True,
    )

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_external_table_avro(snapshot):
    path = "/path/to/files/"

    avro_schema = {
        "fields": [
            {"name": "a", "type": "string"},
            {"name": "b", "type": "int"},
            {"name": "c", "type": "double"},
            {
                "type": "bytes",
                "logicalType": "decimal",
                "precision": 4,
                "scale": 2,
                "name": "d",
            },
        ],
        "name": "my_record",
        "type": "record",
    }

    stmt = ddl.CreateTableAvro(
        "new_table", path, avro_schema, database="foo", can_exist=True
    )

    result = stmt.compile()
    snapshot.assert_match(result, "out.sql")


def test_create_table_parquet(expr, snapshot):
    statement = _create_table("some_table", expr, database="bar", can_exist=False)
    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_no_overwrite(expr, snapshot):
    statement = _create_table("tname", expr, can_exist=True)
    result = statement.compile()
    snapshot.assert_match(result, "out.sql")


def test_avro_other_formats(t, snapshot):
    statement = _create_table("tname", t, format="avro", can_exist=True)
    result = statement.compile()
    snapshot.assert_match(result, "out.sql")

    with pytest.raises(ValueError):
        _create_table("tname", t, format="foo")


def _create_table(table_name, expr, database=None, can_exist=False, format="parquet"):
    ast = ImpalaCompiler.to_ast(expr)
    select = ast.queries[0]
    statement = CTAS(
        table_name,
        select,
        database=database,
        format=format,
        can_exist=can_exist,
    )
    return statement


def _get_select(expr, context=None):
    ast = ImpalaCompiler.to_ast(expr, context)
    select = ast.queries[0]
    context = ast.context
    return select, context
