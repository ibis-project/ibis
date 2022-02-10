from pathlib import Path

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    @staticmethod
    def connect(data_directory: Path):
        path = data_directory / "ibis_testing.ddb"
        return ibis.duckdb.connect(str(path))  # type: ignore
