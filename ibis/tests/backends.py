import abc
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pkg_resources import parse_version

import ibis
import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir


class RoundingConvention:
    @staticmethod
    @abc.abstractmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        """Round a series to `decimals` number of decimal values."""


class RoundAwayFromZero(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        if not decimals:
            return (
                -(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)
            ).astype(np.int64)
        return series.round(decimals=decimals)


class RoundHalfToEven(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        result = series.round(decimals=decimals)
        return result if decimals else result.astype(np.int64)


class Backend(abc.ABC):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    additional_skipped_operations = frozenset()
    supports_divide_by_zero = False
    returned_timestamp_unit = 'us'
    supported_to_timestamp_units = {'s', 'ms', 'us'}
    supports_floating_modulus = True

    def __init__(self, data_directory: Path) -> None:
        self.api  # skips if we can't access the backend
        self.connection = self.connect(data_directory)

    @property
    def name(self) -> str:
        return str(self).lower()

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    @abc.abstractmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        """Return a connection with data loaded from `data_directory`."""

    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(
        series: pd.Series, name: str = 'tmp'
    ) -> pd.Series:
        return series.rename(name)

    @staticmethod
    def greatest(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @staticmethod
    def least(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @property
    def db(self) -> ibis.client.Database:
        return self.connection.database()

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        return self.db.functional_alltypes

    @property
    def batting(self) -> ir.TableExpr:
        return self.db.batting

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.db.awards_players

    @property
    def geo(self) -> Optional[ir.TableExpr]:
        return None

    @property
    def api(self):
        return getattr(ibis, self.name)

    def make_context(
        self, params: Optional[Mapping[ir.ValueExpr, Any]] = None
    ) -> comp.QueryContext:
        return self.api.dialect.make_context(params=params)


class UnorderedComparator:
    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
        return super().assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        columns = list(set(left.columns) & set(right.columns))
        left = left.sort_values(by=columns)
        right = right.sort_values(by=columns)
        return super().assert_frame_equal(left, right, *args, **kwargs)


class Pandas(Backend, RoundHalfToEven):
    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})
    supported_to_timestamp_units = Backend.supported_to_timestamp_units | {
        'ns'
    }
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        return ibis.pandas.connect(
            {
                'functional_alltypes': pd.read_csv(
                    str(data_directory / 'functional_alltypes.csv'),
                    index_col=None,
                    dtype={'bool_col': bool, 'string_col': str},
                    parse_dates=['timestamp_col'],
                    encoding='utf-8',
                ),
                'batting': pd.read_csv(str(data_directory / 'batting.csv')),
                'awards_players': pd.read_csv(
                    str(data_directory / 'awards_players.csv')
                ),
            }
        )


class Csv(Pandas):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        filename = data_directory / 'functional_alltypes.csv'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.csv.connect(data_directory)

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        schema = ibis.schema(
            [
                ('bool_col', dt.boolean),
                ('string_col', dt.string),
                ('timestamp_col', dt.timestamp),
            ]
        )
        return self.connection.table('functional_alltypes', schema=schema)

    @property
    def batting(self) -> ir.TableExpr:
        schema = ibis.schema(
            [
                ('lgID', dt.string),
                ('G', dt.float64),
                ('AB', dt.float64),
                ('R', dt.float64),
                ('H', dt.float64),
                ('X2B', dt.float64),
                ('X3B', dt.float64),
                ('HR', dt.float64),
                ('RBI', dt.float64),
                ('SB', dt.float64),
                ('CS', dt.float64),
                ('BB', dt.float64),
                ('SO', dt.float64),
            ]
        )
        return self.connection.table('batting', schema=schema)

    @property
    def awards_players(self) -> ir.TableExpr:
        schema = ibis.schema(
            [('lgID', dt.string), ('tie', dt.string), ('notes', dt.string)]
        )
        return self.connection.table('awards_players', schema=schema)


class Parquet(Pandas):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        filename = data_directory / 'functional_alltypes.parquet'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.parquet.connect(data_directory)


class HDF5(Pandas):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        filename = data_directory / 'functional_alltypes.h5'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.hdf5.connect(data_directory)


class SQLite(Backend, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = False
    returned_timestamp_unit = 's'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        path = Path(
            os.environ.get(
                'IBIS_TEST_SQLITE_DATABASE', data_directory / 'ibis_testing.db'
            )
        )
        if not path.exists():
            pytest.skip('SQLite testing db {} does not exist'.format(path))
        return ibis.sqlite.connect(str(path))

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = self.db.functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


class Postgres(Backend, RoundHalfToEven):
    # postgres rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = 's'

    @property
    def name(self) -> str:
        return 'postgres'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        user = os.environ.get(
            'IBIS_TEST_POSTGRES_USER', os.environ.get('PGUSER', 'postgres')
        )
        password = os.environ.get(
            'IBIS_TEST_POSTGRES_PASSWORD', os.environ.get('PGPASS', 'postgres')
        )
        host = os.environ.get(
            'IBIS_TEST_POSTGRES_HOST', os.environ.get('PGHOST', 'localhost')
        )
        port = os.environ.get(
            'IBIS_TEST_POSTGRES_PORT', os.environ.get('PGPORT', '5432')
        )
        database = os.environ.get(
            'IBIS_TEST_POSTGRES_DATABASE',
            os.environ.get('PGDATABASE', 'ibis_testing'),
        )
        return ibis.postgres.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

    @property
    def geo(self) -> Optional[ir.TableExpr]:
        if 'geo' in self.db.list_tables():
            return self.db.geo


class OmniSciDB(Backend, RoundAwayFromZero):
    check_dtype = False
    check_names = False
    supports_window_operations = True
    supports_divide_by_zero = False
    supports_floating_modulus = False
    returned_timestamp_unit = 's'
    # Exception: Non-empty LogicalValues not supported yet
    additional_skipped_operations = frozenset(
        {
            ops.Abs,
            ops.Ceil,
            ops.Floor,
            ops.Exp,
            ops.Sign,
            ops.Sqrt,
            ops.Ln,
            ops.Log10,
            ops.Modulus,
        }
    )

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        user = os.environ.get('IBIS_TEST_OMNISCIDB_USER', 'admin')
        password = os.environ.get(
            'IBIS_TEST_OMNISCIDB_PASSWORD', 'HyperInteractive'
        )
        host = os.environ.get('IBIS_TEST_OMNISCIDB_HOST', 'localhost')
        port = os.environ.get('IBIS_TEST_OMNISCIDB_PORT', '6274')
        database = os.environ.get(
            'IBIS_TEST_OMNISCIDB_DATABASE', 'ibis_testing'
        )
        return ibis.omniscidb.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

    @property
    def geo(self) -> Optional[ir.TableExpr]:
        return self.db.geo


class MySQL(Backend, RoundHalfToEven):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)
        # mariadb supports window operations after version 10.2
        # but the sqlalchemy version string looks like:
        # 5.5.5.10.2.12.MariaDB.10.2.12+maria~jessie
        # or 10.4.12.MariaDB.1:10.4.12+maria~bionic
        # example of possible results:
        # https://github.com/sqlalchemy/sqlalchemy/blob/rel_1_3/
        # test/dialect/mysql/test_dialect.py#L244-L268
        con = self.connection
        if 'MariaDB' in str(con.version):
            # we might move this parsing step to the mysql client
            version_detail = con.con.dialect._parse_server_version(
                str(con.version)
            )
            version = (
                version_detail[:3]
                if version_detail[3] == 'MariaDB'
                else version_detail[3:6]
            )
            self.__class__.supports_window_operations = version >= (10, 2)
        elif con.version >= parse_version('8.0'):
            # mysql supports window operations after version 8
            self.__class__.supports_window_operations = True

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        user = os.environ.get('IBIS_TEST_MYSQL_USER', 'ibis')
        password = os.environ.get('IBIS_TEST_MYSQL_PASSWORD', 'ibis')
        host = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
        port = os.environ.get('IBIS_TEST_MYSQL_PORT', 3306)
        database = os.environ.get('IBIS_TEST_MYSQL_DATABASE', 'ibis_testing')
        return ibis.mysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

    @property
    def functional_alltypes(self):
        # BOOLEAN <-> TINYINT(1)
        t = super().functional_alltypes
        return t.mutate(bool_col=t.bool_col == 1)


class Clickhouse(UnorderedComparator, Backend, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supported_to_timestamp_units = {'s'}
    supports_floating_modulus = False

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        user = os.environ.get('IBIS_TEST_CLICKHOUSE_USER', 'default')
        password = os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')
        database = os.environ.get(
            'IBIS_TEST_CLICKHOUSE_DATABASE', 'ibis_testing'
        )
        return ibis.clickhouse.connect(
            host=host,
            port=port,
            password=password,
            database=database,
            user=user,
        )

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = super().functional_alltypes
        return t.mutate(bool_col=t.bool_col == 1)

    @staticmethod
    def greatest(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to greatest'
            )
        return f(*args)

    @staticmethod
    def least(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to least'
            )
        return f(*args)


class BigQuery(UnorderedComparator, Backend, RoundAwayFromZero):
    supports_divide_by_zero = True
    supports_floating_modulus = False
    returned_timestamp_unit = 'us'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        from ibis.bigquery.tests.conftest import connect

        project_id = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')
        if project_id is None:
            pytest.skip(
                'Environment variable GOOGLE_BIGQUERY_PROJECT_ID '
                'not defined'
            )
        elif not project_id:
            pytest.skip(
                'Environment variable GOOGLE_BIGQUERY_PROJECT_ID is empty'
            )
        return connect(project_id, dataset_id='testing')

    @property
    def batting(self) -> ir.TableExpr:
        return None

    @property
    def awards_players(self) -> ir.TableExpr:
        return None


class Impala(UnorderedComparator, Backend, RoundAwayFromZero):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 's'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        from ibis.backends.impala.tests.conftest import IbisTestEnv

        env = IbisTestEnv()
        hdfs_client = ibis.hdfs_connect(
            host=env.nn_host,
            port=env.webhdfs_port,
            auth_mechanism=env.auth_mechanism,
            verify=env.auth_mechanism not in ['GSSAPI', 'LDAP'],
            user=env.webhdfs_user,
        )
        auth_mechanism = env.auth_mechanism
        if auth_mechanism == 'GSSAPI' or auth_mechanism == 'LDAP':
            print("Warning: ignoring invalid Certificate Authority errors")
        return ibis.impala.connect(
            host=env.impala_host,
            port=env.impala_port,
            auth_mechanism=env.auth_mechanism,
            hdfs_client=hdfs_client,
            database='ibis_testing',
        )

    @property
    def batting(self) -> ir.TableExpr:
        return None

    @property
    def awards_players(self) -> ir.TableExpr:
        return None


class Spark(Backend, RoundHalfToEven):
    @staticmethod
    def connect(data_directory):
        from ibis.tests.all.conftest import get_spark_testing_client

        return get_spark_testing_client(data_directory)

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        return self.connection.table('functional_alltypes')

    @property
    def batting(self) -> ir.TableExpr:
        return self.connection.table('batting')

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.connection.table('awards_players')


class PySpark(Backend, RoundAwayFromZero):
    supported_to_timestamp_units = {'s'}

    @staticmethod
    def connect(data_directory):
        from ibis.tests.all.conftest import get_pyspark_testing_client

        return get_pyspark_testing_client(data_directory)

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        return self.connection.table('functional_alltypes')

    @property
    def batting(self) -> ir.TableExpr:
        return self.connection.table('batting')

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.connection.table('awards_players')
