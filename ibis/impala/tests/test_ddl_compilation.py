import pytest

import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala import ddl  # noqa: E402
from ibis.impala.client import build_ast  # noqa: E402
from ibis.impala.compiler import ImpalaDialect  # noqa: E402


pytestmark = pytest.mark.impala


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
def t(mockcon):
    return mockcon.table('functional_alltypes')


def test_select_basics(t):
    name = 'testing123456'

    expr = t.limit(10)
    select, _ = _get_select(expr, ImpalaDialect.make_context())

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
INSERT OVERWRITE foo.`testing123456`
SELECT *
FROM functional_alltypes
LIMIT 10"""
    assert result == expected


def test_load_data_unpartitioned():
    path = '/path/to/data'
    stmt = ddl.LoadData('functional_alltypes', path, database='foo')

    result = stmt.compile()
    expected = ("LOAD DATA INPATH '/path/to/data' "
                "INTO TABLE foo.`functional_alltypes`")
    assert result == expected

    stmt.overwrite = True
    result = stmt.compile()
    expected = ("LOAD DATA INPATH '/path/to/data' "
                "OVERWRITE INTO TABLE foo.`functional_alltypes`")
    assert result == expected


def test_load_data_partitioned():
    path = '/path/to/data'
    part = {'year': 2007, 'month': 7}
    part_schema = ibis.schema([('year', 'int32'), ('month', 'int32')])
    stmt = ddl.LoadData('functional_alltypes', path,
                        database='foo',
                        partition=part,
                        partition_schema=part_schema)

    result = stmt.compile()
    expected = """\
LOAD DATA INPATH '/path/to/data' INTO TABLE foo.`functional_alltypes`
PARTITION (year=2007, month=7)"""
    assert result == expected

    stmt.overwrite = True
    result = stmt.compile()
    expected = """\
LOAD DATA INPATH '/path/to/data' OVERWRITE INTO TABLE foo.`functional_alltypes`
PARTITION (year=2007, month=7)"""
    assert result == expected


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_select_overwrite():
    assert False


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


def test_add_partition(part_schema, table_name):
    stmt = ddl.AddPartition(table_name,
                            {'year': 2007, 'month': 4},
                            part_schema)

    result = stmt.compile()
    expected = 'ALTER TABLE tbl ADD PARTITION (year=2007, month=4)'
    assert result == expected


def test_add_partition_string_key():
    part_schema = ibis.schema([('foo', 'int32'),
                               ('bar', 'string')])
    stmt = ddl.AddPartition('tbl', {'foo': 5, 'bar': 'qux'}, part_schema)

    result = stmt.compile()
    expected = 'ALTER TABLE tbl ADD PARTITION (foo=5, bar="qux")'
    assert result == expected


def test_drop_partition(part_schema, table_name):
    stmt = ddl.DropPartition(table_name,
                             {'year': 2007, 'month': 4},
                             part_schema)

    result = stmt.compile()
    expected = 'ALTER TABLE tbl DROP PARTITION (year=2007, month=4)'
    assert result == expected


def test_add_partition_with_props(part_schema, table_name):
    props = dict(
        location='/users/foo/my-data'
    )
    stmt = ddl.AddPartition(table_name,
                            {'year': 2007, 'month': 4},
                            part_schema, **props)

    result = stmt.compile()
    expected = """\
ALTER TABLE tbl ADD PARTITION (year=2007, month=4)
LOCATION '/users/foo/my-data'"""
    assert result == expected


def test_alter_partition_properties(part_schema, table_name):
    part = {'year': 2007, 'month': 4}

    def _get_ddl_string(props):
        stmt = ddl.AlterPartition(table_name, part,
                                  part_schema, **props)
        return stmt.compile()

    result = _get_ddl_string({'location': '/users/foo/my-data'})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET LOCATION '/users/foo/my-data'"""
    assert result == expected

    result = _get_ddl_string({'format': 'avro'})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET FILEFORMAT AVRO"""
    assert result == expected

    result = _get_ddl_string({'tbl_properties': {
        'bar': 2, 'foo': '1'
    }})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET TBLPROPERTIES (
  'bar'='2',
  'foo'='1'
)"""
    assert result == expected

    result = _get_ddl_string({'serde_properties': {'baz': 3}})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET SERDEPROPERTIES (
  'baz'='3'
)"""
    assert result == expected


def test_alter_table_properties(part_schema, table_name):
    part = {'year': 2007, 'month': 4}

    def _get_ddl_string(props):
        stmt = ddl.AlterPartition(table_name, part,
                                  part_schema, **props)
        return stmt.compile()

    result = _get_ddl_string({'location': '/users/foo/my-data'})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET LOCATION '/users/foo/my-data'"""
    assert result == expected

    result = _get_ddl_string({'format': 'avro'})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET FILEFORMAT AVRO"""
    assert result == expected

    result = _get_ddl_string({'tbl_properties': {
        'bar': 2, 'foo': '1'
    }})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET TBLPROPERTIES (
  'bar'='2',
  'foo'='1'
)"""
    assert result == expected

    result = _get_ddl_string({'serde_properties': {'baz': 3}})
    expected = """\
ALTER TABLE tbl PARTITION (year=2007, month=4)
SET SERDEPROPERTIES (
  'baz'='3'
)"""
    assert result == expected


@pytest.fixture
def expr(t):
    return t[t.bigint_col > 0]


def test_create_external_table_as(mockcon):
    path = '/path/to/table'
    select, _ = _get_select(
        mockcon.table('test1'),
        ImpalaDialect.make_context())
    statement = ddl.CTAS('another_table',
                         select,
                         external=True,
                         can_exist=False,
                         path=path,
                         database='foo')
    result = statement.compile()

    expected = """\
CREATE EXTERNAL TABLE foo.`another_table`
STORED AS PARQUET
LOCATION '{0}'
AS
SELECT *
FROM test1""".format(path)
    assert result == expected


def test_create_table_with_location_compile():
    path = '/path/to/table'
    schema = ibis.schema([('foo', 'string'),
                          ('bar', 'int8'),
                          ('baz', 'int16')])
    statement = ddl.CreateTableWithSchema('another_table', schema,
                                          can_exist=False,
                                          format='parquet',
                                          path=path, database='foo')
    result = statement.compile()

    expected = """\
CREATE TABLE foo.`another_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
STORED AS PARQUET
LOCATION '{0}'""".format(path)
    assert result == expected


def test_create_table_like_parquet():
    directory = '/path/to/'
    path = '/path/to/parquetfile'
    statement = ddl.CreateTableParquet('new_table',
                                       directory,
                                       example_file=path,
                                       can_exist=True,
                                       database='foo')

    result = statement.compile()
    expected = """\
CREATE EXTERNAL TABLE IF NOT EXISTS foo.`new_table`
LIKE PARQUET '{0}'
STORED AS PARQUET
LOCATION '{1}'""".format(path, directory)

    assert result == expected


def test_create_table_parquet_like_other():
    # alternative to "LIKE PARQUET"
    directory = '/path/to/'
    example_table = 'db.other'

    statement = ddl.CreateTableParquet('new_table',
                                       directory,
                                       example_table=example_table,
                                       can_exist=True,
                                       database='foo')

    result = statement.compile()
    expected = """\
CREATE EXTERNAL TABLE IF NOT EXISTS foo.`new_table`
LIKE {0}
STORED AS PARQUET
LOCATION '{1}'""".format(example_table, directory)

    assert result == expected


def test_create_table_parquet_with_schema():
    directory = '/path/to/'

    schema = ibis.schema([('foo', 'string'),
                          ('bar', 'int8'),
                          ('baz', 'int16')])

    statement = ddl.CreateTableParquet('new_table',
                                       directory,
                                       schema=schema,
                                       external=True,
                                       can_exist=True,
                                       database='foo')

    result = statement.compile()
    expected = """\
CREATE EXTERNAL TABLE IF NOT EXISTS foo.`new_table`
(`foo` string,
 `bar` tinyint,
 `baz` smallint)
STORED AS PARQUET
LOCATION '{0}'""".format(directory)

    assert result == expected


def test_create_table_delimited():
    path = '/path/to/files/'
    schema = ibis.schema([('a', 'string'),
                          ('b', 'int32'),
                          ('c', 'double'),
                          ('d', 'decimal(12, 2)')])

    stmt = ddl.CreateTableDelimited('new_table', path, schema,
                                    delimiter='|',
                                    escapechar='\\',
                                    lineterminator='\0',
                                    database='foo',
                                    can_exist=True)

    result = stmt.compile()
    expected = """\
CREATE EXTERNAL TABLE IF NOT EXISTS foo.`new_table`
(`a` string,
 `b` int,
 `c` double,
 `d` decimal(12, 2))
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
ESCAPED BY '\\'
LINES TERMINATED BY '\0'
LOCATION '{0}'""".format(path)
    assert result == expected


def test_create_external_table_avro():
    path = '/path/to/files/'

    avro_schema = {
        'fields': [
            {'name': 'a', 'type': 'string'},
            {'name': 'b', 'type': 'int'},
            {'name': 'c', 'type': 'double'},
            {"type": "bytes",
             "logicalType": "decimal",
             "precision": 4,
             "scale": 2,
             'name': 'd'}
        ],
        'name': 'my_record',
        'type': 'record'
    }

    stmt = ddl.CreateTableAvro('new_table', path, avro_schema,
                               database='foo', can_exist=True)

    result = stmt.compile()
    expected = """\
CREATE EXTERNAL TABLE IF NOT EXISTS foo.`new_table`
STORED AS AVRO
LOCATION '%s'
TBLPROPERTIES (
  'avro.schema.literal'='{
  "fields": [
    {
      "name": "a",
      "type": "string"
    },
    {
      "name": "b",
      "type": "int"
    },
    {
      "name": "c",
      "type": "double"
    },
    {
      "logicalType": "decimal",
      "name": "d",
      "precision": 4,
      "scale": 2,
      "type": "bytes"
    }
  ],
  "name": "my_record",
  "type": "record"
}'
)""" % path
    assert result == expected


def test_create_table_parquet(expr):
    statement = _create_table('some_table', expr,
                              database='bar',
                              can_exist=False)
    result = statement.compile()

    expected = """\
CREATE TABLE bar.`some_table`
STORED AS PARQUET
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
STORED AS PARQUET
AS
SELECT *
FROM functional_alltypes
WHERE `bigint_col` > 0"""
    assert result == expected


def test_avro_other_formats(t):
    statement = _create_table('tname', t, format='avro', can_exist=True)
    result = statement.compile()
    expected = """\
CREATE TABLE IF NOT EXISTS `tname`
STORED AS AVRO
AS
SELECT *
FROM functional_alltypes"""
    assert result == expected

    with pytest.raises(ValueError):
        _create_table('tname', t, format='foo')


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_partition_by():
    assert False


def _create_table(table_name, expr, database=None, can_exist=False,
                  format='parquet'):
    ast = build_ast(expr, ImpalaDialect.make_context())
    select = ast.queries[0]
    statement = ddl.CTAS(table_name, select,
                         database=database,
                         format=format,
                         can_exist=can_exist)
    return statement


def _get_select(expr, context):
    ast = build_ast(expr, context)
    select = ast.queries[0]
    context = ast.context
    return select, context
