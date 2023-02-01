from __future__ import annotations

import os
from pathlib import Path

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

DREMIO_URI = os.environ.get("IBIS_TEST_DREMIO_URI")
DREMIO_USER = os.environ.get("IBIS_TEST_DREMIO_USER")
DREMIO_PASS = os.environ.get("IBIS_TEST_DREMIO_PASS")


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = False
    supports_json = False
    supports_structs = False
    supports_window_operations = False

    @staticmethod
    def connect(data_directory: Path):
        return ibis.dremio.connect(
            uri=DREMIO_URI,
            username=DREMIO_USER,
            password=DREMIO_PASS,
            context="@dremio",
        )
