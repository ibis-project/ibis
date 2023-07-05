from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from pathlib import Path

pa = pytest.importorskip("pyarrow")


class TestConf(BackendTest, RoundAwayFromZero):
    # check_names = False
    # supports_divide_by_zero = True
    # returned_timestamp_unit = 'ns'
    supports_structs = False
    supports_json = False
    supports_arrays = False

    @staticmethod
    def connect(data_directory: Path):
        client = ibis.datafusion.connect({})
        client.register(
            data_directory / "parquet" / 'functional_alltypes.parquet',
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
