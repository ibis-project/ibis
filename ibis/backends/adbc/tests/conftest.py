from __future__ import annotations

import base64
import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

    @staticmethod
    def _load_data(
        data_dir,
        script_dir,
        database: str = "ibis_testing",
        **_: Any,
    ) -> None:
        """Load test data into a ADBC backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        adbc = pytest.importorskip("pyarrow.flight_sql")

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def connect(data_directory: Path) -> BaseBackend:

        flight_password = os.environ["FLIGHT_PASSWORD"]
        authorization_header = f"Basic {str(base64.b64encode(bytes(f'flight_username:{flight_password}', encoding='utf-8')), encoding='utf-8')}"

        con = ibis.adbc.connect(
            dialect="duckdb",
            uri="grpc+tls://localhost:31337",
            db_kwargs={
                "arrow.flight.sql.authorization_header": authorization_header,
                "arrow.flight.sql.client_option.disable_server_verification": "true",
            },
        )

        return con
