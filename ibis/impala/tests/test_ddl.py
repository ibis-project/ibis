from posixpath import join as pjoin

import pytest

import ibis
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.util as util

from ibis.tests.util import assert_equal

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala.compat import HS2Error  # noqa: E402

pytestmark = pytest.mark.impala


def test_create_exists_view(con, temp_view):
    tmp_name = temp_view
    assert not con.exists_table(tmp_name)

    expr = con.table('functional_alltypes').group_by('string_col').size()

    con.create_view(tmp_name, expr)
    assert con.exists_table(tmp_name)

    # just check it works for now
    expr2 = con.table(tmp_name)
    assert expr2.execute() is not None


def test_drop_non_empty_database(con, alltypes, temp_table_db):
    temp_database, temp_table = temp_table_db
    con.create_table(temp_table, alltypes, database=temp_database)
    assert con.exists_table(temp_table, database=temp_database)

    with pytest.raises(com.IntegrityError):
        con.drop_database(temp_database)


def test_create_database_with_location(con, tmp_dir, hdfs):
    base = pjoin(tmp_dir, util.guid())
    name = '__ibis_test_{}'.format(util.guid())
    tmp_path = pjoin(base, name)

    con.create_database(name, path=tmp_path)
    try:
        assert hdfs.exists(base)
    finally:
        try:
            con.drop_database(name)
        finally:
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
    assert hdfs.exists(tmp_path)


def test_drop_table_not_exist(con):
    non_existent_table = 'ibis_table_{}'.format(util.guid())
    with pytest.raises(Exception):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


def test_truncate_table(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    table_name = temp_table
    con.create_table(table_name, obj=expr)

    try:
        con.truncate_table(table_name)
    except HS2Error as e:
        if 'AnalysisException' in e.args[0]:
            pytest.skip('TRUNCATE not available in this version of Impala')

    t = con.table(table_name)
    nrows = t.count().execute()
    assert not nrows


def test_truncate_table_expression(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    table_name = temp_table
    con.create_table(table_name, obj=expr)
    t = con.table(table_name)
    t.truncate()
    nrows = t.count().execute()
    assert not nrows


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


@pytest.fixture
def created_view(con, alltypes):
    name = util.guid()
    expr = alltypes.limit(10)
    con.create_view(name, expr)
    return name


def test_drop_view(con, alltypes, created_view):
    con.drop_view(created_view)
    assert not con.exists_table(created_view)


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
    assert con.exists_table(name, database=tmp_db)

    expr = table.r_name.value_counts()
    expr.execute()

    assert table.count().execute() == 5

    df = table.execute()
    assert len(df) == 5


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
    # we don't install futures on windows in CI and we can't run this test
    # there anyway so we import here
    import concurrent.futures
    from concurrent.futures import as_completed

    def limit_10(i, hdfs_path):
        t = con.parquet_file(hdfs_path)
        return t.sort_by(t.r_regionkey).limit(1, offset=i).execute()

    nthreads = 4
    hdfs_path = pjoin(test_data_dir, 'parquet/tpch_region')

    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as e:
        futures = [e.submit(limit_10, i, hdfs_path) for i in range(nthreads)]
    assert all(map(len, (future.result() for future in as_completed(futures))))


def test_access_kudu_table(kudu_table):
    assert kudu_table.columns == ['a']
    assert kudu_table['a'].type() == dt.string
