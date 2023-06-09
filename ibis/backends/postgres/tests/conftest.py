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
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import init_database
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest, ServiceSpec

if TYPE_CHECKING:
    from pathlib import Path

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


class TestConf(ServiceBackendTest, RoundHalfToEven):
    # postgres rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = 's'
    supports_structs = False

    @classmethod
    def service_spec(cls, data_dir: Path) -> ServiceSpec:
        return ServiceSpec(
            name=cls.name(),
            data_volume="/data",
            files=data_dir.joinpath("csv").glob("*.csv"),
        )

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
            init_database(
                url=sa.engine.make_url(
                    f"postgresql://{user}:{password}@{host}:{port:d}/{database}"
                ),
                database=database,
                schema=schema,
                isolation_level="AUTOCOMMIT",
                recreate=False,
            )

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
def alltypes_sqla(con, alltypes):
    name = alltypes.op().name
    return con._get_sqla_table(name)


@pytest.fixture(scope='module')
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.postgres import Backend

    context = Backend.compiler.make_context()
    return lambda expr: Backend.compiler.translator_class(expr, context).get_result()
