from __future__ import annotations

from pathlib import Path

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

pa = pytest.importorskip("pyarrow")


class TestConf(BackendTest, RoundAwayFromZero):
    # check_names = False
    # additional_skipped_operations = frozenset({ops.StringSQLLike})
    # supports_divide_by_zero = True
    # returned_timestamp_unit = 'ns'
    native_bool = False
    supports_structs = False
    supports_json = False

    @staticmethod
    def connect(data_directory: Path):
        # can be various types:
        #   pyarrow.RecordBatch
        #   parquet file path
        #   csv file path
        client = ibis.datafusion.connect({})
        client.register(
            data_directory / 'functional_alltypes.csv',
            table_name='functional_alltypes',
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
        client.register(data_directory / 'batting.csv', table_name='batting')
        client.register(
            data_directory / 'awards_players.csv', table_name='awards_players'
        )
        client.register(data_directory / 'diamonds.csv', table_name='diamonds')
        return client

    @property
    def functional_alltypes(self) -> ir.Table:
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
