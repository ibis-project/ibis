from unittest import mock

import pandas as pd


def patch_execute(con):
    return mock.patch.object(con, 'raw_sql', wraps=con.raw_sql)


def test_invalidate_metadata(backend_instance, test_data_db):
    with patch_execute(backend_instance) as ex_mock:
        backend_instance.invalidate_metadata()
        ex_mock.assert_called_with('INVALIDATE METADATA')

    backend_instance.invalidate_metadata('functional_alltypes')
    t = backend_instance.table('functional_alltypes')
    t.invalidate_metadata()

    with patch_execute(backend_instance) as ex_mock:
        backend_instance.invalidate_metadata(
            'functional_alltypes', database=test_data_db
        )
        ex_mock.assert_called_with(
            'INVALIDATE METADATA {}.`{}`'.format(
                test_data_db, 'functional_alltypes'
            )
        )


def test_refresh(backend_instance, test_data_db):
    tname = 'functional_alltypes'
    with patch_execute(backend_instance) as ex_mock:
        backend_instance.refresh(tname)
        ex_cmd = f'REFRESH {test_data_db}.`{tname}`'
        ex_mock.assert_called_with(ex_cmd)

    t = backend_instance.table(tname)
    with patch_execute(backend_instance) as ex_mock:
        t.refresh()
        ex_cmd = f'REFRESH {test_data_db}.`{tname}`'
        ex_mock.assert_called_with(ex_cmd)


def test_describe_formatted(backend_instance, test_data_db):
    from ibis.backends.impala.metadata import TableMetadata

    t = backend_instance.table('functional_alltypes')
    with patch_execute(backend_instance) as ex_mock:
        desc = t.describe_formatted()
        ex_mock.assert_called_with(
            'DESCRIBE FORMATTED '
            '{}.`{}`'.format(test_data_db, 'functional_alltypes'),
            results=True,
        )
        assert isinstance(desc, TableMetadata)


def test_show_files(backend_instance, test_data_db):
    t = backend_instance.table('functional_alltypes')
    qualified_name = '{}.`{}`'.format(test_data_db, 'functional_alltypes')
    with patch_execute(backend_instance) as ex_mock:
        desc = t.files()
        ex_mock.assert_called_with(
            f'SHOW FILES IN {qualified_name}', results=True
        )
        assert isinstance(desc, pd.DataFrame)


def test_table_column_stats(backend_instance, test_data_db):
    t = backend_instance.table('functional_alltypes')

    qualified_name = '{}.`{}`'.format(test_data_db, 'functional_alltypes')
    with patch_execute(backend_instance) as ex_mock:
        desc = t.stats()
        ex_mock.assert_called_with(
            f'SHOW TABLE STATS {qualified_name}', results=True
        )
        assert isinstance(desc, pd.DataFrame)

    with patch_execute(backend_instance) as ex_mock:
        desc = t.column_stats()
        ex_mock.assert_called_with(
            f'SHOW COLUMN STATS {qualified_name}', results=True
        )
        assert isinstance(desc, pd.DataFrame)
