import gc

import ibis

from posixpath import join as pjoin
import pytest

import ibis.common as com
import ibis.expr.types as ir
import ibis.util as util
from ibis.tests.util import assert_equal

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala import ddl  # noqa: E402
from ibis.impala.compat import HS2Error, ImpylaError  # noqa: E402
from ibis.impala.client import build_ast  # noqa: E402
from ibis.impala.compiler import ImpalaDialect  # noqa: E402


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


@pytest.fixture
def path_uuid():
    return 'change-location-{0}'.format(util.guid())


@pytest.fixture
def table(con, tmp_db, tmp_dir, path_uuid):
    table_name = 'table_{}'.format(util.guid())
    fake_path = pjoin(tmp_dir, path_uuid)
    schema = ibis.schema([('foo', 'string'), ('bar', 'int64')])
    con.create_table(table_name,
                     database=tmp_db,
                     schema=schema,
                     format='parquet',
                     external=True,
                     location=fake_path)
    try:
        yield con.table(table_name, database=tmp_db)
    finally:
        con.drop_table(table_name, database=tmp_db)


def test_list_databases(con):
    assert con.list_databases()


def test_list_tables(con, test_data_db):
    assert con.list_tables(database=test_data_db)
    assert con.list_tables(like='*nat*', database=test_data_db)


def test_set_database(con_no_db, test_data_db):
    # create new connection with no default db set
    # TODO: set test_data_db to None
    with pytest.raises(Exception):
        con_no_db.table('functional_alltypes')
    con_no_db.set_database(test_data_db)
    assert con_no_db.table('functional_alltypes') is not None


def test_tables_robust_to_set_database(con, temp_database):
    table = con.table('functional_alltypes')

    con.set_database(temp_database)

    # it still works!
    n = 10
    df = table.limit(n).execute()
    assert len(df) == n


def test_exists_table(con):
    assert con.exists_table('functional_alltypes')
    assert not con.exists_table('foobarbaz_{}'.format(util.guid()))


def text_exists_table_with_database(
    con, alltypes, test_data_db, temp_table, temp_database
):
    table_name = temp_table
    tmp_db = test_data_db
    con.create_table(table_name, alltypes, database=tmp_db)

    assert con.exists_table(table_name, database=tmp_db)
    assert not con.exists_table(table_name, database=temp_database)


def test_create_exists_view(con, temp_view):
    tmp_name = temp_view
    assert not con.exists_table(tmp_name)

    expr = con.table('functional_alltypes').group_by('string_col').size()

    con.create_view(tmp_name, expr)
    assert con.exists_table(tmp_name)

    # just check it works for now
    expr2 = con.table(tmp_name)
    expr2.execute()


def test_drop_non_empty_database(con, alltypes, temp_table_db, temp_view_db):
    temp_database, temp_table = temp_table_db
    _, temp_view = temp_view_db
    con.create_table(temp_table, alltypes, database=temp_database)

    # Has a view, too
    con.create_view(temp_view, alltypes, database=temp_database)

    with pytest.raises(com.IntegrityError):
        con.drop_database(temp_database)


def test_create_database_with_location(con, tmp_dir, hdfs):
    base = pjoin(tmp_dir, util.guid())
    name = '__ibis_test_{0}'.format(util.guid())
    tmp_path = pjoin(base, name)

    con.create_database(name, path=tmp_path)
    try:
        assert hdfs.exists(base)
    finally:
        con.drop_database(name)
        hdfs.rmdir(base)


def test_create_table_with_location_execute(
    con, hdfs, tmp_dir, alltypes, test_data_db, temp_table
):
    base = pjoin(tmp_dir, util.guid())
    name = 'test_{}'.format(util.guid())
    tmp_path = pjoin(base, name)

    expr = alltypes
    table_name = temp_table

    con.create_table(
        table_name, obj=expr, location=tmp_path, database=test_data_db)
    # self.temp_tables.append('{}.{}'.format(test_data_db, table_name))
    assert hdfs.exists(tmp_path)


def test_drop_table_not_exist(con):
    non_existent_table = 'ibis_table_{}'.format(util.guid())
    with pytest.raises(Exception):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


def test_truncate_table(con, alltypes, temp_table):
    expr = alltypes.limit(50)

    table_name = temp_table
    con.create_table(table_name, obj=expr)

    try:
        con.truncate_table(table_name)
    except HS2Error as e:
        if 'AnalysisException' in e.args[0]:
            pytest.skip('TRUNCATE not available in this version of Impala')

    result = con.table(table_name).execute()
    assert len(result) == 0


def test_truncate_table_expression(con, alltypes, temp_table):
    expr = alltypes.limit(5)

    table_name = temp_table
    con.create_table(table_name, obj=expr)
    t = con.table(table_name)
    t.truncate()
    assert len(t.execute()) == 0


def test_ctas_from_table_expr(con, alltypes, temp_table_db):
    expr = alltypes
    db, table_name = temp_table_db

    con.create_table(table_name, expr, database=db)


def test_create_empty_table(con, temp_table):
    schema = ibis.schema([('a', 'string'),
                          ('b', 'timestamp'),
                          ('c', 'decimal(12, 8)'),
                          ('d', 'double')])

    table_name = temp_table
    con.create_table(table_name, schema=schema)

    result_schema = con.get_schema(table_name)
    assert_equal(result_schema, schema)

    assert con.table(table_name).execute().empty


def test_insert_table(con, alltypes, temp_table, test_data_db):
    expr = alltypes
    table_name = temp_table
    db = test_data_db

    con.create_table(table_name, expr.limit(0), database=db)
    # self.temp_tables.append('.'.join((db, table_name)))

    con.insert(table_name, expr.limit(10), database=db)

    # check using ImpalaTable.insert
    t = con.table(table_name, database=db)
    t.insert(expr.limit(10))

    sz = t.count()
    assert sz.execute() == 20

    # Overwrite and verify only 10 rows now
    t.insert(expr.limit(10), overwrite=True)
    assert sz.execute() == 10


def test_insert_validate_types(con, alltypes, test_data_db, temp_table):
    # GH #235
    table_name = temp_table
    db = test_data_db

    expr = alltypes
    con.create_table(
        table_name,
        schema=expr['tinyint_col', 'int_col', 'string_col'].schema(),
        database=db
    )
    # self.temp_tables.append('.'.join((db, table_name)))

    t = con.table(table_name, database=db)

    to_insert = expr[expr.tinyint_col, expr.smallint_col.name('int_col'),
                     expr.string_col]
    t.insert(to_insert.limit(10))

    to_insert = expr[expr.tinyint_col,
                     expr.smallint_col.cast('int32').name('int_col'),
                     expr.string_col]
    t.insert(to_insert.limit(10))

    to_insert = expr[expr.tinyint_col,
                     expr.bigint_col.name('int_col'),
                     expr.string_col]

    limit_expr = to_insert.limit(10)
    with pytest.raises(com.IbisError):
        t.insert(limit_expr)


def test_compute_stats(con):
    t = con.table('functional_alltypes')

    t.compute_stats()
    t.compute_stats(incremental=True)

    con.compute_stats('functional_alltypes')


def test_drop_table_or_view(con, db, temp_table, temp_view):
    t = db.functional_alltypes

    tname = temp_table
    con.create_table(tname, t.limit(10))

    vname = temp_view
    con.create_view(vname, t.limit(10))


def test_rename_table(con, temp_database):
    tmp_db = temp_database

    orig_name = 'tmp_rename_test'
    con.create_table(orig_name, con.table('tpch_region'))
    table = con.table(orig_name)

    old_name = table.name

    new_name = 'rename_test'
    renamed = table.rename(new_name, database=tmp_db)
    renamed.execute()

    t = con.table(new_name, database=tmp_db)
    assert_equal(renamed, t)

    assert table.name == old_name


def test_change_location(con, table, tmp_dir, path_uuid):
    old_loc = table.metadata().location

    new_path = pjoin(tmp_dir, 'new-path')
    table.alter(location=new_path)

    new_loc = table.metadata().location
    assert new_loc == old_loc.replace(path_uuid, 'new-path')


def test_change_properties(con, table):
    props = {'foo': '1', 'bar': '2'}

    table.alter(tbl_properties=props)
    tbl_props = table.metadata().tbl_properties
    for k, v in props.items():
        assert v == tbl_props[k]

    table.alter(serde_properties=props)
    serde_props = table.metadata().serde_properties
    for k, v in props.items():
        assert v == serde_props[k]


def test_change_format(con, table):
    table.alter(format='avro')

    meta = table.metadata()
    assert 'Avro' in meta.hive_format


def test_cleanup_tmp_table_on_gc(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')
    table = con.parquet_file(hdfs_path)
    name = table.op().name
    table = None
    gc.collect()
    _assert_table_not_exists(con, name)


def test_persist_parquet_file_with_name(con, test_data_dir, temp_table_db):
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    tmp_db, name = temp_table_db
    schema = ibis.schema([('r_regionkey', 'int16'),
                          ('r_name', 'string'),
                          ('r_comment', 'string')])
    con.parquet_file(hdfs_path, schema=schema, name=name, database=tmp_db,
                     persist=True)
    gc.collect()

    # table still exists
    con.table(name, database=tmp_db)


def test_query_avro(con, test_data_dir, tmp_db):
    hdfs_path = pjoin(test_data_dir, 'avro/tpch_region_avro')

    avro_schema = {
        "fields": [
            {"type": ["int", "null"], "name": "R_REGIONKEY"},
            {"type": ["string", "null"], "name": "R_NAME"},
            {"type": ["string", "null"], "name": "R_COMMENT"}],
        "type": "record",
        "name": "a"
    }

    table = con.avro_file(hdfs_path, avro_schema, database=tmp_db)

    name = table.op().name
    assert name.startswith('{}.'.format(tmp_db))

    # table exists
    con.table(name)

    expr = table.r_name.value_counts()
    expr.execute()

    assert table.count().execute() == 5

    df = table.execute()
    assert len(df) == 5


def test_query_parquet_file_with_schema(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    schema = ibis.schema([('r_regionkey', 'int16'),
                          ('r_name', 'string'),
                          ('r_comment', 'string')])

    table = con.parquet_file(hdfs_path, schema=schema)

    name = table.op().name

    # table exists
    con.table(name)

    expr = table.r_name.value_counts()
    expr.execute()

    assert table.count().execute() == 5


def test_query_parquet_file_like_table(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    ex_schema = ibis.schema([('r_regionkey', 'int16'),
                             ('r_name', 'string'),
                             ('r_comment', 'string')])

    table = con.parquet_file(hdfs_path, like_table='tpch_region')

    assert_equal(table.schema(), ex_schema)


def test_query_parquet_infer_schema(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')
    table = con.parquet_file(hdfs_path)

    # NOTE: the actual schema should have an int16, but bc this is being
    # inferred from a parquet file, which has no notion of int16, the
    # inferred schema will have an int32 instead.
    ex_schema = ibis.schema([('r_regionkey', 'int32'),
                             ('r_name', 'string'),
                             ('r_comment', 'string')])

    assert_equal(table.schema(), ex_schema)


def test_create_table_persist_fails_if_called_twice(
    con, temp_table, test_data_dir
):
    tname = temp_table

    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')
    con.parquet_file(hdfs_path, name=tname, persist=True)

    with pytest.raises(HS2Error):
        con.parquet_file(hdfs_path, name=tname, persist=True)


def test_create_table_reserved_identifier(con):
    table_name = 'distinct'
    expr = con.table('functional_alltypes')
    expected = expr.count().execute()
    con.create_table(table_name, expr)
    try:
        result = con.table(table_name).count().execute()
    except Exception:
        raise
    else:
        assert result == expected
    finally:
        con.drop_table(table_name)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_query_text_file_regex():
    assert False


def test_query_delimited_file_directory(con, test_data_dir, tmp_db):
    hdfs_path = pjoin(test_data_dir, 'csv')

    schema = ibis.schema([('foo', 'string'),
                          ('bar', 'double'),
                          ('baz', 'int8')])
    name = 'delimited_table_test1'
    table = con.delimited_file(hdfs_path, schema, name=name, database=tmp_db,
                               delimiter=',')

    expr = (table
            [table.bar > 0]
            .group_by('foo')
            .aggregate([table.bar.sum().name('sum(bar)'),
                        table.baz.sum().name('mean(baz)')]))
    assert expr.execute() is not None


def test_varchar_char_support(temp_char_table):
    assert isinstance(temp_char_table['group1'], ir.StringValue)
    assert isinstance(temp_char_table['group2'], ir.StringValue)


def test_temp_table_concurrency(con, test_data_dir):
    pytest.skip('Cannot get this test to run under pytest')

    from threading import Thread, Lock
    import gc
    nthreads = 4

    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    lock = Lock()

    results = []

    def do_something():
        t = con.parquet_file(hdfs_path)

        with lock:
            t.limit(10).execute()
            t = None
            gc.collect()
            results.append(True)

    threads = []
    for i in range(nthreads):
        t = Thread(target=do_something)
        t.start()
        threads.append(t)

    [x.join() for x in threads]

    assert results == [True] * nthreads


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


def _assert_table_not_exists(con, table_name, database=None):
    if database is not None:
        tname = '.'.join((database, table_name))
    else:
        tname = table_name

    try:
        con.table(tname)
    except ImpylaError:
        pass


def _ensure_drop(con, table_name, database=None):
    con.drop_table(table_name, database=database, force=True)
    _assert_table_not_exists(con, table_name, database=database)
