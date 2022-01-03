from pathlib import Path

import pyarrow as pa
import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    # check_names = False
    # additional_skipped_operations = frozenset({ops.StringSQLLike})
    # supports_divide_by_zero = True
    # returned_timestamp_unit = 'ns'
    bool_is_int = True

    @staticmethod
    def connect(data_directory: Path):
        # can be various types:
        #   pyarrow.RecordBatch
        #   parquet file path
        #   csv file path
        client = ibis.datafusion.connect({})
        client.register_csv(
            name='functional_alltypes',
            path=data_directory / 'functional_alltypes.csv',
            schema=pa.schema(
                [
                    ('index', 'int64'),
                    ('Unnamed 0', 'int64'),
                    ('id', 'int64'),
                    ('bool_col', 'int8'),
                    ('tinyint_col', 'int8'),
                    ('smallint_col', 'int16'),
                    ('int_col', 'int32'),
                    ('bigint_col', 'int64'),
                    ('float_col', 'float32'),
                    ('double_col', 'float64'),
                    ('date_string_col', 'string'),
                    ('string_col', 'string'),
                    ('timestamp_col', 'string'),
                    ('year', 'int64'),
                    ('month', 'int64'),
                ]
            ),
        )
        client.register_csv(
            name='batting', path=data_directory / 'batting.csv'
        )
        client.register_csv(
            name='awards_players', path=data_directory / 'awards_players.csv'
        )
        return client

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = self.connection.table('functional_alltypes')
        return t.mutate(
            bool_col=t.bool_col == 1,
            timestamp_col=t.timestamp_col.cast('timestamp'),
        )


@pytest.fixture(scope='session')
def client(data_directory):
    return TestConf.connect(data_directory)


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table("functional_alltypes")


@pytest.fixture(scope='session')
def alltypes_df(alltypes):
    return alltypes.execute()
