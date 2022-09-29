from pathlib import Path

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import array_types

pl = pytest.importorskip("polars")


class TestConf(BackendTest, RoundAwayFromZero):
    bool_is_int = True
    supports_structs = False
    supports_json = False
    reduction_tolerance = 1e-3

    @staticmethod
    def connect(data_directory: Path):
        client = ibis.polars.connect({})
        client.register_csv(
            name='functional_alltypes',
            path=data_directory / 'functional_alltypes.csv',
            dtypes={
                'index': pl.Int64,
                'Unnamed 0': pl.Int64,
                'id': pl.Int64,
                'bool_col': pl.Int64,
                'tinyint_col': pl.Int64,
                'smallint_col': pl.Int64,
                'int_col': pl.Int32,
                'bigint_col': pl.Int64,
                'float_col': pl.Float32,
                'double_col': pl.Float64,
                'date_string_col': pl.Utf8,
                'string_col': pl.Utf8,
                'timestamp_col': pl.Datetime,
                'year': pl.Int64,
                'month': pl.Int64,
            },
        )
        client.register_csv(name='batting', path=data_directory / 'batting.csv')
        client.register_csv(
            name='awards_players', path=data_directory / 'awards_players.csv'
        )
        client.register_pandas(name='array_types', df=array_types)

        return client

    @property
    def functional_alltypes(self) -> ir.Table:
        table = self.connection.table('functional_alltypes')
        return table.mutate(
            bool_col=table.bool_col.cast('bool'),
            tinyint_col=table.tinyint_col.cast('int8'),
            smallint_col=table.smallint_col.cast('int16'),
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
