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
from typing import TYPE_CHECKING

import pytest

import ibis
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

PG_USER = os.environ.get(
    "IBIS_TEST_POSTGRES_USER", os.environ.get("PGUSER", "postgres")
)
PG_PASS = os.environ.get(
    "IBIS_TEST_POSTGRES_PASSWORD", os.environ.get("PGPASSWORD", "postgres")
)
PG_HOST = os.environ.get(
    "IBIS_TEST_POSTGRES_HOST", os.environ.get("PGHOST", "localhost")
)
PG_PORT = os.environ.get("IBIS_TEST_POSTGRES_PORT", os.environ.get("PGPORT", 5432))
IBIS_TEST_POSTGRES_DB = os.environ.get(
    "IBIS_TEST_POSTGRES_DATABASE", os.environ.get("PGDATABASE", "ibis_testing")
)


class TestConf(ServiceBackendTest):
    # postgres rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = "s"
    supports_structs = False
    rounding_method = "half_to_even"
    service_name = "postgres"
    deps = ("psycopg2",)

    driver_supports_multiple_statements = True

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.postgres.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_POSTGRES_DB,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def geotable(con):
    return con.table("geo")


@pytest.fixture(scope="module")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="module")
def gdf(geotable):
    return geotable.execute()


@pytest.fixture(scope="module")
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.postgres import Backend

    context = Backend.compiler.make_context()
    return lambda expr: Backend.compiler.translator_class(expr, context).get_result()
