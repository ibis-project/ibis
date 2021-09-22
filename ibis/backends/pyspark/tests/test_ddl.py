import os
from posixpath import join as pjoin

import pyspark as ps
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis.tests.util import assert_equal


def test_create_exists_view(client, alltypes, temp_view):
    tmp_name = temp_view
    assert tmp_name not in client.list_tables()

    expr = alltypes.group_by('string_col').size()

    client.create_view(tmp_name, expr, temporary=True)
    assert tmp_name in client.list_tables()

    # just check it works for now
    expr2 = client.table(tmp_name)
    assert expr2.execute() is not None


def test_drop_non_empty_database(client, alltypes, temp_table_db):
    temp_database, temp_table = temp_table_db
    client.create_table(temp_table, alltypes, database=temp_database)
    assert temp_table in client.list_tables(database=temp_database)

    with pytest.raises(ps.sql.utils.AnalysisException):
        client.drop_database(temp_database)


def test_create_database_with_location(client, tmp_dir):
    base = pjoin(tmp_dir, util.guid())
    name = f'__ibis_test_{util.guid()}'
    tmp_path = pjoin(base, name)

    client.create_database(name, path=tmp_path)
    try:
        assert os.path.exists(base)
    finally:
        try:
            client.drop_database(name, force=True)
        finally:
            os.rmdir(base)


def test_drop_table_not_exist(client):
    non_existent_table = f'ibis_table_{util.guid()}'
    with pytest.raises(Exception):
        client.drop_table(non_existent_table)
    client.drop_table(non_existent_table, force=True)


@pytest.mark.xfail(
    reason='Moved from the legacy spark backend, '
    'but not working in the pyspark one #2879'
)
def test_truncate_table(client, alltypes, temp_table):
    expr = alltypes.limit(1)

    table_name = temp_table
    client.create_table(table_name, obj=expr)

    client.truncate_table(table_name)

    t = client.table(table_name)
    nrows = t.count().execute()
    assert not nrows


@pytest.mark.xfail(
    reason='Moved from the legacy spark backend, '
    'but not working in the pyspark one #2879'
)
def test_truncate_table_expression(client, alltypes, temp_table):
    expr = alltypes.limit(1)

    table_name = temp_table
    client.create_table(table_name, obj=expr)
    t = client.table(table_name)
    t.truncate()
    nrows = t.count().execute()
    assert not nrows


def test_ctas_from_table_expr(client, alltypes, temp_table_db):
    expr = alltypes
    db, table_name = temp_table_db

    client.create_table(table_name, expr, database=db)


def test_create_empty_table(client, temp_table):
    schema = ibis.schema(
        [
            ('a', 'string'),
            ('b', 'timestamp'),
            ('c', 'decimal(12, 8)'),
            ('d', 'double'),
        ]
    )

    table_name = temp_table
    client.create_table(table_name, schema=schema)

    result_schema = client.get_schema(table_name)
    assert_equal(result_schema, schema)

    assert client.table(table_name).execute().empty


@pytest.mark.xfail(
    reason='Moved from the legacy spark backend, '
    'but not working in the pyspark one #2879'
)
def test_insert_table(client, alltypes, temp_table, test_data_db):
    expr = alltypes
    table_name = temp_table
    db = test_data_db

    client.create_table(table_name, expr.limit(0), database=db)

    client.insert(table_name, expr.limit(10), database=db)

    # check using SparkTable.insert
    t = client.table(table_name, database=db)
    t.insert(expr.limit(10))

    sz = t.count()
    assert sz.execute() == 20

    # Overwrite and verify only 10 rows now
    t.insert(expr.limit(10), overwrite=True)
    assert sz.execute() == 10


def test_insert_validate_types(client, alltypes, test_data_db, temp_table):
    table_name = temp_table
    db = test_data_db

    expr = alltypes
    client.create_table(
        table_name,
        schema=expr['tinyint_col', 'int_col', 'string_col'].schema(),
        database=db,
    )

    t = client.table(table_name, database=db)

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


def test_compute_stats(client, alltypes):
    name = 'functional_alltypes_table'
    try:
        client.create_table(name, alltypes)
        t = client.table(name)
        t.compute_stats()
        t.compute_stats(noscan=True)
        client.compute_stats(name)
    finally:
        client.drop_table(name, force=True)


@pytest.fixture
def created_view(client, alltypes):
    name = util.guid()
    expr = alltypes.limit(10)
    client.create_view(name, expr, temporary=True)
    return name


def test_drop_view(client, alltypes, created_view):
    client.drop_view(created_view)
    assert created_view not in client.list_tables()


def test_rename_table(client, alltypes):
    orig_name = 'tmp_rename_test'
    client.create_table(orig_name, alltypes)
    table = client.table(orig_name)

    old_name = table.name

    new_name = 'rename_test'
    renamed = table.rename(new_name)
    renamed.execute()

    t = client.table(new_name)
    assert_equal(renamed, t)

    assert table.name == old_name


@pytest.fixture
def table(client, temp_database):
    table_name = f'table_{util.guid()}'
    schema = ibis.schema([('foo', 'string'), ('bar', 'int64')])
    client.create_table(
        table_name, database=temp_database, schema=schema, format='parquet'
    )
    try:
        yield client.table(table_name, database=temp_database)
    finally:
        client.drop_table(table_name, database=temp_database)


def test_change_properties(client, table):
    props = {'foo': '1', 'bar': '2'}

    table.alter(tbl_properties=props)
    tbl_props_rows = client.raw_sql(
        f"show tblproperties {table.name}"
    ).fetchall()
    for row in tbl_props_rows:
        key = row.key
        value = row.value
        assert value == props[key]


@pytest.mark.xfail(
    reason='Moved from the legacy spark backend, '
    'but not working in the pyspark one #2879'
)
def test_create_table_reserved_identifier(client, alltypes):
    table_name = 'distinct'
    expr = alltypes
    expected = expr.count().execute()
    client.create_table(table_name, expr)
    try:
        result = client.table(table_name).count().execute()
    except Exception:
        raise
    else:
        assert result == expected
    finally:
        client.drop_table(table_name)


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


def test_schema_from_csv(client, awards_players_filename):
    schema = client._schema_from_csv(awards_players_filename)
    assert schema.equals(awards_players_schema)


@pytest.mark.xfail(
    reason='Moved from the legacy spark backend, '
    'but not working in the pyspark one #2879'
)
def test_create_table_or_temp_view_from_csv(client, awards_players_filename):
    client._create_table_or_temp_view_from_csv(
        'awards', awards_players_filename
    )
    table = client.table('awards')
    assert table.schema().equals(awards_players_schema)
    assert table.count().execute() == 6078
