from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import array_types, struct_types, win

if TYPE_CHECKING:
    from pathlib import Path

pl = pytest.importorskip("polars")


class TestConf(BackendTest, RoundAwayFromZero):
    supports_structs = True
    supports_json = False
    reduction_tolerance = 1e-3

    @staticmethod
    def connect(data_directory: Path):
        client = ibis.polars.connect({})
        client.register(
            data_directory / 'parquet' / 'functional_alltypes.parquet',
            table_name='functional_alltypes',
        )
        client.register(
            data_directory / "parquet" / 'batting.parquet', table_name='batting'
        )
        client.register(
            data_directory / "parquet" / 'awards_players.parquet',
            table_name='awards_players',
        )
        client.register(
            data_directory / "parquet" / 'diamonds.parquet', table_name='diamonds'
        )
        client.register(array_types, table_name='array_types')
        client.register(struct_types, table_name='struct')
        client.register(win, table_name="win")

        return client


@pytest.fixture(scope='session')
def client(data_directory):
    return TestConf.connect(data_directory)


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table("functional_alltypes")


@pytest.fixture(scope='session')
def alltypes_df(alltypes):
    return alltypes.execute()
