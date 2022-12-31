from __future__ import annotations

from pathlib import Path

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import array_types, struct_types, win

pl = pytest.importorskip("polars")


class TestConf(BackendTest, RoundAwayFromZero):
    supports_structs = True
    supports_json = False
    reduction_tolerance = 1e-3

    @staticmethod
    def connect(data_directory: Path):
        client = ibis.polars.connect({})
        client.register(
            data_directory / 'functional_alltypes.csv',
            table_name='functional_alltypes',
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
        client.register(data_directory / 'batting.csv', table_name='batting')
        client.register(
            data_directory / 'awards_players.csv', table_name='awards_players'
        )
        client.register(data_directory / 'diamonds.csv', table_name='diamonds')
        client.register(array_types, table_name='array_types')
        client.register(struct_types, table_name='struct')
        client.register(win, table_name="win")

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
