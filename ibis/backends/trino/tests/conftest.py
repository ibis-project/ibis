import os
from pathlib import Path
from typing import Any, Generator

import pytest

import ibis
from ibis.backends.conftest import _random_identifier
from ibis.backends.tests.base import BackendTest, RoundHalfToEven

TRINO_USER = os.environ.get(
    'IBIS_TEST_TRINO_USER', os.environ.get('TRINO_USER', 'user')
)
TRINO_PASS = os.environ.get(
    'IBIS_TEST_TRINO_PASSWORD', os.environ.get('TRINO_PASSWORD', '')
)
TRINO_HOST = os.environ.get(
    'IBIS_TEST_TRINO_HOST', os.environ.get('TRINO_HOST', 'localhost')
)
TRINO_PORT = os.environ.get('IBIS_TEST_TRINO_PORT', os.environ.get('TRINO_PORT', 8080))
IBIS_TEST_TRINO_DB = os.environ.get(
    'IBIS_TEST_TRINO_DATABASE',
    os.environ.get('TRINO_DATABASE', 'postgresql'),
)


class TestConf(BackendTest, RoundHalfToEven):
    # trino rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = 's'
    supports_structs = False

    @staticmethod
    def _load_data(data_dir: Path, script_dir: Path, **_: Any) -> None:
        """Load test data into a Trino backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        from ibis.backends.postgres.tests.conftest import TestConf

        TestConf._load_data(data_dir, script_dir, port=5433)

    @staticmethod
    def connect(data_directory: Path):
        return ibis.trino.connect(
            host=TRINO_HOST,
            port=TRINO_PORT,
            user=TRINO_USER,
            password=TRINO_PASS,
            database=IBIS_TEST_TRINO_DB,
            schema="public",
        )

    @property
    def batting(self):
        b = self.connection.table("batting")
        b = b.relabel(
            {
                "yearid": "yearID",
                "lgid": "lgID",
                "playerid": "playerID",
                "teamid": "teamID",
                "rbi": "RBI",
                "g": "G",
            }
        )
        return b

    @property
    def awards_players(self):
        a = self.connection.table("awards_players")
        a = a.relabel(
            {
                "yearid": "yearID",
                "lgid": "lgID",
                "playerid": "playerID",
                "awardid": "awardID",
            }
        )
        return a


@pytest.fixture(scope='session')
def con(tmp_path_factory, data_directory, script_directory, worker_id):
    return TestConf.load_data(
        data_directory,
        script_directory,
        tmp_path_factory,
        worker_id,
    ).connect(data_directory)


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def geotable(con):
    return con.table('geo')


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='module')
def gdf(geotable):
    return geotable.execute()


@pytest.fixture(scope='module')
def at(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope='module')
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.trino import Backend

    context = Backend.compiler.make_context()
    return lambda expr: (Backend.compiler.translator_class(expr, context).get_result())


@pytest.fixture
def temp_table(con) -> Generator[str, None, None]:
    """Return a temporary table name."""
    name = _random_identifier('table')
    try:
        yield name
    finally:
        con.drop_table(name, force=True)
