import os
from posixpath import join as pjoin

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis.tests.util import assert_equal

pytestmark = pytest.mark.spark
ps = pytest.importorskip('pyspark')


def test_create_exists_view(con, alltypes, temp_view):
    tmp_name = temp_view
    assert not con.exists_table(tmp_name)

    expr = alltypes.group_by('string_col').size()

    con.create_view(tmp_name, expr, temporary=True)
    assert con.exists_table(tmp_name)

    # just check it works for now
    expr2 = con.table(tmp_name)
    assert expr2.execute() is not None


def test_drop_non_empty_database(con, alltypes, temp_table_db):
    temp_database, temp_table = temp_table_db
    con.create_table(temp_table, alltypes, database=temp_database)
    assert con.exists_table(temp_table, database=temp_database)

    with pytest.raises(ps.sql.utils.AnalysisException):
        con.drop_database(temp_database)


def test_create_database_with_location(con, tmp_dir):
    base = pjoin(tmp_dir, util.guid())
    name = '__ibis_test_{}'.format(util.guid())
    tmp_path = pjoin(base, name)

    con.create_database(name, path=tmp_path)
    try:
        assert os.path.exists(base)
    finally:
        try:
            con.drop_database(name, force=True)
        finally:
            os.rmdir(base)


def test_drop_table_not_exist(con):
    non_existent_table = 'ibis_table_{}'.format(util.guid())
    with pytest.raises(Exception):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


def test_truncate_table(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    table_name = temp_table
    con.create_table(table_name, obj=expr)

    con.truncate_table(table_name)

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
    schema = ibis.schema(
        [
            ('a', 'string'),
            ('b', 'timestamp'),
            ('c', 'decimal(12, 8)'),
            ('d', 'double'),
        ]
    )

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

    # check using SparkTable.insert
    t = con.table(table_name, database=db)
    t.insert(expr.limit(10))

    sz = t.count()
    assert sz.execute() == 20

    # Overwrite and verify only 10 rows now
    t.insert(expr.limit(10), overwrite=True)
    assert sz.execute() == 10


def test_insert_validate_types(con, alltypes, test_data_db, temp_table):
    table_name = temp_table
    db = test_data_db

    expr = alltypes
    con.create_table(
        table_name,
        schema=expr['tinyint_col', 'int_col', 'string_col'].schema(),
        database=db,
    )

    t = con.table(table_name, database=db)

    to_insert = expr[
        expr.tinyint_col, expr.smallint_col.name('int_col'), expr.string_col
    ]
    t.insert(to_insert.limit(10))

    to_insert = expr[
        expr.tinyint_col,
        expr.smallint_col.cast('int32').name('int_col'),
        expr.string_col,
    ]
    t.insert(to_insert.limit(10))

    to_insert = expr[
        expr.tinyint_col, expr.bigint_col.name('int_col'), expr.string_col
    ]

    limit_expr = to_insert.limit(10)
    with pytest.raises(com.IbisError):
        t.insert(limit_expr)


def test_compute_stats(con, alltypes):
    name = 'functional_alltypes_table'
    try:
        con.create_table(name, alltypes)
        t = con.table(name)
        t.compute_stats()
        t.compute_stats(noscan=True)
        con.compute_stats(name)
    finally:
        con.drop_table(name, force=True)


@pytest.fixture
def created_view(con, alltypes):
    name = util.guid()
    expr = alltypes.limit(10)
    con.create_view(name, expr, temporary=True)
    return name


def test_drop_view(con, alltypes, created_view):
    con.drop_view(created_view)
    assert not con.exists_table(created_view)


def test_rename_table(con, alltypes):
    orig_name = 'tmp_rename_test'
    con.create_table(orig_name, alltypes)
    table = con.table(orig_name)

    old_name = table.name

    new_name = 'rename_test'
    renamed = table.rename(new_name)
    renamed.execute()

    t = con.table(new_name)
    assert_equal(renamed, t)

    assert table.name == old_name


@pytest.fixture
def table(con, temp_database):
    table_name = 'table_{}'.format(util.guid())
    schema = ibis.schema([('foo', 'string'), ('bar', 'int64')])
    con.create_table(
        table_name, database=temp_database, schema=schema, format='parquet'
    )
    try:
        yield con.table(table_name, database=temp_database)
    finally:
        con.drop_table(table_name, database=temp_database)


def test_change_properties(con, table):
    props = {'foo': '1', 'bar': '2'}

    table.alter(tbl_properties=props)
    tbl_props_rows = con.raw_sql(
        "show tblproperties {}".format(table.name), results=True
    ).fetchall()
    for row in tbl_props_rows:
        key = row.key
        value = row.value
        assert value == props[key]


def test_create_table_reserved_identifier(con, alltypes):
    table_name = 'distinct'
    expr = alltypes
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


@pytest.fixture(scope='session')
def awards_players_filename(data_directory):
    return str(data_directory / 'awards_players.csv')


awards_players_schema = ibis.schema(
    [
        ('playerID', 'string'),
        ('awardID', 'string'),
        ('yearID', 'int32'),
        ('lgID', 'string'),
        ('tie', 'string'),
        ('notes', 'string'),
    ]
)


def test_schema_from_csv(con, awards_players_filename):
    schema = con._schema_from_csv(awards_players_filename)
    assert schema.equals(awards_players_schema)


def test_create_table_or_temp_view_from_csv(con, awards_players_filename):
    con._create_table_or_temp_view_from_csv('awards', awards_players_filename)
    table = con.table('awards')
    assert table.schema().equals(awards_players_schema)
    assert table.count().execute() == 6078
