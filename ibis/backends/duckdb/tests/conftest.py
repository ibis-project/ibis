from functools import lru_cache
from pathlib import Path

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

    @staticmethod
    @lru_cache(maxsize=None)
    def connect(data_directory: Path):
        path = data_directory / "ibis_testing.ddb"
        return ibis.duckdb.connect(str(path))  # type: ignore
