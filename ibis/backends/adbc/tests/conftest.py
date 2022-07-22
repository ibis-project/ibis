from __future__ import annotations

import os
from pathlib import Path

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_structs = False

    @staticmethod
    def connect(data_directory: Path):
        path = Path(
            os.environ.get(
                'IBIS_TEST_SQLITE_DATABASE', data_directory / 'ibis_testing.db'
            )
        )
        return ibis.adbc.connect(
            driver="adbc_driver_sqlite",
            entrypoint="AdbcSqliteDriverInit",
            db_args=dict(
                filename=str(path),
            ),
            dialect="sqlite",
        )

    @property
    def functional_alltypes(self) -> ir.Table:
        t = super().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))
