import os
import pytest
import pandas as pd
import pandas.util.testing as tm

from contextlib import contextmanager

import ibis
import ibis.common as com
import ibis.expr.operations as ops
from ibis.compat import Path
from ibis.impala.tests.common import IbisTestEnv as ImpalaEnv


class Backend(object):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    additional_skipped_operations = frozenset()

    def __init__(self, data_directory):
        self.connection = self.connect(data_directory)

    @property
    def name(self):
        return str(self).lower()

    def __str__(self):
        return self.__class__.__name__

    def connect(self, data_directory):
        raise NotImplementedError

    @contextmanager
    def skip_unsupported(self):
        try:
            yield
        except (com.OperationNotDefinedError, com.UnsupportedBackendType) as e:
            pytest.skip('{} using {}'.format(e, str(self)))

    def assert_series_equal(self, *args, **kwargs):
        kwargs.setdefault('check_dtype', self.check_dtype)
        kwargs.setdefault('check_names', self.check_names)
        tm.assert_series_equal(*args, **kwargs)

    def assert_frame_equal(self, *args, **kwargs):
        tm.assert_frame_equal(*args, **kwargs)

    def default_series_rename(self, series, name='tmp'):
        return series.rename(name)

    def functional_alltypes(self):
        return self.connection.database().functional_alltypes

    def functional_alltypes_df(self):
        return self.functional_alltypes().execute()


class UnorderedSeriesComparator(object):

    def assert_series_equal(self, left, right, *args, **kwargs):
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
        return super(UnorderedSeriesComparator, self).assert_series_equal(
            left, right, *args, **kwargs)


class Csv(Backend):
    check_names = False

    def connect(self, data_directory):
        filename = data_directory / 'functional_alltypes.csv'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.csv.connect(data_directory)

    def functional_alltypes(self):
        schema = ibis.schema([
            ('bool_col', 'boolean'),
            ('string_col', 'string'),
            ('timestamp_col', 'timestamp')
        ])
        return self.connection.table('functional_alltypes', schema=schema)
        # return table.mutate(date_col=table.timestamp_col.date())


class Parquet(Backend):
    check_names = False

    def connect(self, data_directory):
        filename = data_directory / 'functional_alltypes.parquet'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.parquet.connect(data_directory)


class Pandas(Backend):
    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})

    def connect(self, data_directory):
        filename = data_directory / 'functional_alltypes.csv'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))

        return ibis.pandas.connect({
            'functional_alltypes': pd.read_csv(
                filename,
                index_col=None,
                dtype={
                    'string_col': str,
                    'bool_col': bool,
                },
                parse_dates=[
                    'timestamp_col'
                ]
            )
        })


class SQLite(Backend):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = False
    check_dtype = False

    def connect(self, data_directory):
        path = os.environ.get('IBIS_TEST_SQLITE_DATABASE',
                              data_directory / 'ibis_testing.db')
        path = Path(path)
        if not path.exists():
            pytest.skip('SQLite testing db {} does not exist'.format(path))
        return ibis.sqlite.connect(str(path))


class Postgres(Backend):

    def connect(self, data_directory):
        user = os.environ.get('IBIS_TEST_POSTGRES_USER',
                              os.environ.get('PGUSER', 'postgres'))
        password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD',
                                  os.environ.get('PGPASS'))
        host = os.environ.get('IBIS_TEST_POSTGRES_HOST',
                              os.environ.get('PGHOST', 'localhost'))
        database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
                                  os.environ.get('PGDATABASE', 'ibis_testing'))
        return ibis.postgres.connect(host=host, user=user, password=password,
                                     database=database)

    # def functional_alltypes_df(self):
    #     df = super(Postgres, self).functional_alltypes_df()
    #     return df.assign(string_col=df.string_col.str.encode('utf-8'))


class Clickhouse(Backend):
    check_dtype = False

    def connect(self, data_directory):
        host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        user = os.environ.get('IBIS_TEST_CLICKHOUSE_USER', 'default')
        password = os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')
        database = os.environ.get('IBIS_TEST_CLICKHOUSE_DATABASE',
                                  'ibis_testing')
        return ibis.clickhouse.connect(host=host, port=port, password=password,
                                       database=database, user=user)

    def functional_alltypes(self):
        table = self.connection.database().functional_alltypes
        return table.mutate(bool_col=table.bool_col == 1)
        # date_col=table.timestamp_col.date())


class Impala(UnorderedSeriesComparator, Backend):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False

    @classmethod
    def connect(cls, data_directory):
        env = ImpalaEnv()
        hdfs_client = ibis.hdfs_connect(
            host=env.nn_host,
            port=env.webhdfs_port,
            auth_mechanism=env.auth_mechanism,
            verify=env.auth_mechanism not in ['GSSAPI', 'LDAP'],
            user=env.webhdfs_user
        )
        auth_mechanism = env.auth_mechanism
        if auth_mechanism == 'GSSAPI' or auth_mechanism == 'LDAP':
            print("Warning: ignoring invalid Certificate Authority errors")
        return ibis.impala.connect(
            host=env.impala_host,
            port=env.impala_port,
            auth_mechanism=env.auth_mechanism,
            hdfs_client=hdfs_client,
            database='ibis_testing'
        )
