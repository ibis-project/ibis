from unittest import mock

import pandas as pd
import pytest

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

pytestmark = pytest.mark.impala


def patch_execute(con):
    return mock.patch.object(con, '_execute', wraps=con._execute)


def test_invalidate_metadata(con, test_data_db):
    with patch_execute(con) as ex_mock:
        con.invalidate_metadata()
        ex_mock.assert_called_with('INVALIDATE METADATA')

    con.invalidate_metadata('functional_alltypes')
    t = con.table('functional_alltypes')
    t.invalidate_metadata()

    with patch_execute(con) as ex_mock:
        con.invalidate_metadata('functional_alltypes', database=test_data_db)
        ex_mock.assert_called_with(
            'INVALIDATE METADATA {}.`{}`'.format(
                test_data_db, 'functional_alltypes'
            )
        )


def test_refresh(con, test_data_db):
    tname = 'functional_alltypes'
    with patch_execute(con) as ex_mock:
        con.refresh(tname)
        ex_cmd = 'REFRESH {0}.`{1}`'.format(test_data_db, tname)
        ex_mock.assert_called_with(ex_cmd)

    t = con.table(tname)
    with patch_execute(con) as ex_mock:
        t.refresh()
        ex_cmd = 'REFRESH {0}.`{1}`'.format(test_data_db, tname)
        ex_mock.assert_called_with(ex_cmd)


def test_describe_formatted(con, test_data_db):
    from ibis.backends.impala.metadata import TableMetadata

    t = con.table('functional_alltypes')
    with patch_execute(con) as ex_mock:
        desc = t.describe_formatted()
        ex_mock.assert_called_with(
            'DESCRIBE FORMATTED '
            '{0}.`{1}`'.format(test_data_db, 'functional_alltypes'),
            results=True,
        )
        assert isinstance(desc, TableMetadata)


def test_show_files(con, test_data_db):
    t = con.table('functional_alltypes')
    qualified_name = '{0}.`{1}`'.format(test_data_db, 'functional_alltypes')
    with patch_execute(con) as ex_mock:
        desc = t.files()
        ex_mock.assert_called_with(
            'SHOW FILES IN {0}'.format(qualified_name), results=True
        )
        assert isinstance(desc, pd.DataFrame)


def test_table_column_stats(con, test_data_db):
    t = con.table('functional_alltypes')

    qualified_name = '{0}.`{1}`'.format(test_data_db, 'functional_alltypes')
    with patch_execute(con) as ex_mock:
        desc = t.stats()
        ex_mock.assert_called_with(
            'SHOW TABLE STATS {0}'.format(qualified_name), results=True
        )
        assert isinstance(desc, pd.DataFrame)

    with patch_execute(con) as ex_mock:
        desc = t.column_stats()
        ex_mock.assert_called_with(
            'SHOW COLUMN STATS {0}'.format(qualified_name), results=True
        )
        assert isinstance(desc, pd.DataFrame)
