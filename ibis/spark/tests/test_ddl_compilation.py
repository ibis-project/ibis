import pytest

import ibis

from ibis.spark.compiler import (  # noqa: E402, isort:skip
    SparkDialect,
    build_ast,
)
from ibis.spark import ddl  # noqa: E402, isort:skip


pytestmark = pytest.mark.spark


def test_drop_table_compile():
    statement = ddl.DropTable('foo', database='bar', must_exist=True)
    query = statement.compile()
    expected = "DROP TABLE bar.`foo`"
    assert query == expected

    statement = ddl.DropTable('foo', database='bar', must_exist=False)
    query = statement.compile()
    expected = "DROP TABLE IF EXISTS bar.`foo`"
    assert query == expected


@pytest.fixture
def t(client):
    return client.table('functional_alltypes')


def test_select_basics(t):
    name = 'testing123456'

    expr = t.limit(10)
    select, _ = _get_select(expr, SparkDialect.make_context())

    stmt = ddl.InsertSelect(name, select, database='foo')
    result = stmt.compile()

    expected = """\
INSERT INTO foo.`testing123456`
SELECT *
FROM functional_alltypes
LIMIT 10"""
    assert result == expected

    stmt = ddl.InsertSelect(name, select, database='foo', overwrite=True)
    result = stmt.compile()

    expected = """\
INSERT OVERWRITE TABLE foo.`testing123456`
SELECT *
FROM functional_alltypes
LIMIT 10"""
    assert result == expected


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_select_overwrite():
    assert False


# TODO implement
@pytest.mark.xfail
def test_cache_table_pool_name():
    statement = ddl.CacheTable('foo', database='bar')
    query = statement.compile()
    expected = "ALTER TABLE bar.`foo` SET CACHED IN 'default'"
    assert query == expected

    statement = ddl.CacheTable('foo', database='bar', pool='my_pool')
    query = statement.compile()
    expected = "ALTER TABLE bar.`foo` SET CACHED IN 'my_pool'"
    assert query == expected


@pytest.fixture
def part_schema():
    return ibis.schema([('year', 'int32'), ('month', 'int32')])


@pytest.fixture
def table_name():
    return 'tbl'


def test_alter_table_properties(part_schema, table_name):
    stmt = ddl.AlterTable(
        'tbl',
        {'bar': 2, 'foo': '1'},
    )
    result = stmt.compile()
    expected = """\
ALTER TABLE tbl SET
TBLPROPERTIES (
  'bar'='2',
  'foo'='1'
)"""
    assert result == expected


@pytest.fixture
def expr(t):
    return t[t.bigint_col > 0]


def test_create_external_table_as(mockcon):
    path = '/path/to/table'
    select, _ = _get_select(
        mockcon.table('test1'), SparkDialect.make_context()
    )
    statement = ddl.CTAS(
        'another_table',
        select,
        can_exist=False,
        path=path,
        database='foo',
    )
    result = statement.compile()

    expected = """\
CREATE TABLE foo.`another_table`
USING PARQUET
LOCATION '{0}'
AS
SELECT *
FROM test1""".format(
        path
    )
    assert result == expected


def test_create_table_with_location_compile():
    path = '/path/to/table'
    schema = ibis.schema(
        [('foo', 'string'), ('bar', 'int8'), ('baz', 'int16')]
    )
    statement = ddl.CreateTableWithSchema(
        'another_table',
        schema,
        can_exist=False,
        format='parquet',
        path=path,
        database='foo',
    )
    result = statement.compile()

    expected = """\
CREATE TABLE foo.`another_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
USING PARQUET
LOCATION '{0}'""".format(
        path
    )
    assert result == expected


def test_create_table_parquet_with_schema():
    directory = '/path/to/'

    schema = ibis.schema(
        [('foo', 'string'), ('bar', 'int8'), ('baz', 'int16')]
    )

    statement = ddl.CreateTableWithSchema(
        'new_table',
        schema,
        database='foo',
        format='parquet',
        can_exist=True,
        path=directory,
    )

    result = statement.compile()
    expected = """\
CREATE TABLE IF NOT EXISTS foo.`new_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
USING PARQUET
LOCATION '{0}'""".format(
        directory
    )

    assert result == expected


def test_create_table_parquet(expr):
    statement = _create_table(
        'some_table', expr, database='bar', can_exist=False
    )
    result = statement.compile()

    expected = """\
CREATE TABLE bar.`some_table`
USING PARQUET
AS
SELECT *
FROM functional_alltypes
WHERE `bigint_col` > 0"""
    assert result == expected


def test_no_overwrite(expr):
    statement = _create_table('tname', expr, can_exist=True)
    result = statement.compile()

    expected = """\
CREATE TABLE IF NOT EXISTS `tname`
USING PARQUET
AS
SELECT *
FROM functional_alltypes
WHERE `bigint_col` > 0"""
    assert result == expected


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_partition_by():
    assert False


def _create_table(
    table_name, expr, database=None, can_exist=False, format='parquet'
):
    ast = build_ast(expr, SparkDialect.make_context())
    select = ast.queries[0]
    statement = ddl.CTAS(
        table_name,
        select,
        database=database,
        format=format,
        can_exist=can_exist,
    )
    return statement


def _get_select(expr, context):
    ast = build_ast(expr, context)
    select = ast.queries[0]
    context = ast.context
    return select, context
