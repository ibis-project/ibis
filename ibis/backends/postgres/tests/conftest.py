from __future__ import annotations

# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from typing import Any, Generator

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import TEST_TABLES, _random_identifier, init_database
from ibis.backends.tests.base import BackendTest, RoundHalfToEven

PG_USER = os.environ.get(
    'IBIS_TEST_POSTGRES_USER', os.environ.get('PGUSER', 'postgres')
)
PG_PASS = os.environ.get(
    'IBIS_TEST_POSTGRES_PASSWORD', os.environ.get('PGPASSWORD', 'postgres')
)
PG_HOST = os.environ.get(
    'IBIS_TEST_POSTGRES_HOST', os.environ.get('PGHOST', 'localhost')
)
PG_PORT = os.environ.get('IBIS_TEST_POSTGRES_PORT', os.environ.get('PGPORT', 5432))
IBIS_TEST_POSTGRES_DB = os.environ.get(
    'IBIS_TEST_POSTGRES_DATABASE', os.environ.get('PGDATABASE', 'ibis_testing')
)


class TestConf(BackendTest, RoundHalfToEven):
    # postgres rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = 's'
    supports_structs = False

    @staticmethod
    def _load_data(
        data_dir: Path,
        script_dir: Path,
        user: str = PG_USER,
        password: str = PG_PASS,
        host: str = PG_HOST,
        port: int = PG_PORT,
        database: str = IBIS_TEST_POSTGRES_DB,
        **_: Any,
    ) -> None:
        """Load test data into a PostgreSQL backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        with open(script_dir / 'schema' / 'postgresql.sql') as schema:
            engine = init_database(
                url=sa.engine.make_url(
                    f"postgresql://{user}:{password}@{host}:{port:d}/{database}"
                ),
                database=database,
                schema=schema,
                isolation_level='AUTOCOMMIT',
                recreate=False,
            )

        tables = list(TEST_TABLES) + ['geo']
        with engine.begin() as con, con.connection.cursor() as cur:
            for table in tables:
                # Here we insert rows using COPY table FROM STDIN, using
                # psycopg2's `copy_expert` API.
                #
                # We could use DataFrame.to_sql(method=callable), but that
                # incurs an unnecessary round trip and requires more code: the
                # `data_iter` argument would have to be turned back into a CSV
                # before being passed to `copy_expert`.
                sql = f"COPY {table} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
                with data_dir.joinpath(f'{table}.csv').open('r') as file:
                    cur.copy_expert(sql=sql, file=file)

            con.exec_driver_sql("VACUUM FULL ANALYZE")

    @staticmethod
    def connect(data_directory: Path):
        return ibis.postgres.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_POSTGRES_DB,
        )


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
    from ibis.backends.postgres import Backend

    context = Backend.compiler.make_context()
    return lambda expr: (Backend.compiler.translator_class(expr, context).get_result())


@pytest.fixture
def temp_table(con) -> Generator[str, None, None]:
    """Return a temporary table name.

    Parameters
    ----------
    con : ibis.postgres.PostgreSQLClient

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    name = _random_identifier('table')
    try:
        yield name
    finally:
        con.drop_table(name, force=True)
