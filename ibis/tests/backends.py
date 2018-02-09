import os
import six
import pytest
import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.operations as ops
from ibis.compat import Path, parse_version
from ibis.impala.tests.common import IbisTestEnv as ImpalaEnv


class Backend(object):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    additional_skipped_operations = frozenset()

    def __init__(self, data_directory):
        try:
            # check that the backend is available
            getattr(ibis, self.name)
        except AttributeError:
            pytest.skip('Backend {} cannot be imported'.format(self.name))
        else:
            self.connection = self.connect(data_directory)

    @property
    def name(self):
        return str(self).lower()

    def __str__(self):
        return self.__class__.__name__

    def connect(self, data_directory):
        raise NotImplementedError

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

    def batting(self):
        return self.connection.database().batting

    def awards_players(self):
        return self.connection.database().awards_players

    @classmethod
    def make_context(cls, params=None):
        module_name = cls.__name__.lower()
        module = getattr(ibis, module_name, None)
        if module is None:
            pytest.skip('Unable to import backend {!r}'.format(module_name))
        return module.dialect.make_context(params=params)


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
        return ibis.pandas.connect({
            'functional_alltypes': pd.read_csv(
                data_directory / 'functional_alltypes.csv',
                index_col=None,
                dtype={'bool_col': bool, 'string_col': six.text_type},
                parse_dates=['timestamp_col'],
                encoding='utf-8'
            ),
            'batting': pd.read_csv(
                data_directory / 'batting.csv'
            ),
            'awards_players': pd.read_csv(
                data_directory / 'awards_players.csv'
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

    def functional_alltypes(self):
        t = self.connection.database().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


class Postgres(Backend):

    def connect(self, data_directory):
        user = os.environ.get('IBIS_TEST_POSTGRES_USER',
                              os.environ.get('PGUSER', 'postgres'))
        password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD',
                                  os.environ.get('PGPASS', 'postgres'))
        host = os.environ.get('IBIS_TEST_POSTGRES_HOST',
                              os.environ.get('PGHOST', 'localhost'))
        database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
                                  os.environ.get('PGDATABASE', 'ibis_testing'))
        return ibis.postgres.connect(host=host, user=user, password=password,
                                     database=database)


class MySQL(Backend):
    check_dtype = False
    supports_window_operations = False

    def connect(self, data_directory):
        user = os.environ.get('IBIS_TEST_MYSQL_USER', 'ibis')
        password = os.environ.get('IBIS_TEST_MYSQL_PASSWORD', 'ibis')
        host = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
        database = os.environ.get('IBIS_TEST_MYSQL_DATABASE', 'ibis_testing')
        con = ibis.mysql.connect(host=host, user=user, password=password,
                                 database=database)

        # mariadb supports window operations after version 10.2
        # but the sqlalchemy version string looks like:
        # 5.5.5.10.2.12.MariaDB.10.2.12+maria~jessie
        if 'MariaDB' in str(con.version):
            # we might move this parsing step to the mysql client
            version = tuple(map(int, str(con.version).split('.')[7:9]))
            if version >= (10, 2):
                self.supports_window_operations = True
        elif con.version >= parse_version('8.0'):
            # mysql supports window operations after version 8
            self.supports_window_operations = True

        return con

    def functional_alltypes(self):
        # BOOLEAN <-> TINYINT(1)
        t = self.connection.database().functional_alltypes
        return t.mutate(bool_col=t.bool_col == 1)


class Clickhouse(Backend):
    check_dtype = False
    supports_window_operations = False

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
        t = self.connection.database().functional_alltypes
        return t.mutate(bool_col=t.bool_col == 1)


class BigQuery(UnorderedSeriesComparator, Backend):

    def connect(self, data_directory):
        ga = pytest.importorskip('google.auth')

        project_id = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')
        if project_id is None:
            pytest.skip('Environment variable GOOGLE_BIGQUERY_PROJECT_ID '
                        'not defined')
        elif not project_id:
            pytest.skip('Environment variable GOOGLE_BIGQUERY_PROJECT_ID '
                        'is empty')

        dataset_id = 'testing'
        try:
            return ibis.bigquery.connect(project_id, dataset_id)
        except ga.exceptions.DefaultCredentialsError:
            pytest.skip('no bigquery credentials found')


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
