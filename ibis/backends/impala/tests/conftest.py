from __future__ import annotations

import contextlib
import itertools
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import ibis
import ibis.expr.types as ir
from ibis import options, util
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.tests.expr.mocks import MockBackend

if TYPE_CHECKING:
    from collections.abc import Iterable


class TestConf(BackendTest):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False
    supports_divide_by_zero = True
    returned_timestamp_unit = "s"
    supports_structs = False
    supports_json = False
    force_sort = True
    deps = ("impala",)
    service_name = "impala"

    @property
    def test_files(self) -> Iterable[Path]:
        return [self.data_dir.joinpath("impala")]

    def preload(self):
        env = IbisTestEnv()

        for directory in self.test_files:
            raw_data_path = f"{env.test_data_dir}/{directory.name}"
            # copy from local to impala volume
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "cp",
                    str(directory),
                    f"{self.service_name}:{raw_data_path}",
                ],
                check=True,
            )

        IBIS_HOME = Path(__file__).absolute().parents[4]
        cwd = str(IBIS_HOME / "ci" / "udf")
        subprocess.run(["cmake", ".", "-G", "Ninja"], cwd=cwd, check=True)
        subprocess.run(["ninja"], cwd=cwd, check=True)
        build_dir = IBIS_HOME / "ci" / "udf" / "build"

        subprocess.run(
            [
                "docker",
                "compose",
                "cp",
                str(build_dir),
                f"{self.service_name}:{env.test_data_dir}/udf",
            ],
            check=True,
        )

    def _load_data(self, **_: Any) -> None:
        """Load test data into a backend."""
        con = self.connection
        database = "ibis_testing"

        con.create_database(database, force=True)
        con.raw_sql(f"USE {database}")

        (parquet,) = self.test_files

        # container path to data
        prefix = "/user/hive/warehouse/impala/parquet"
        for dir in parquet.joinpath("parquet").glob("*"):
            con.drop_table(dir.name, database=database, force=True)

            location = f"{prefix}/{dir.name}"
            first_file = next(
                itertools.chain(dir.rglob("*.parq"), dir.rglob("*.parquet"))
            )

            con.parquet_file(
                location,
                name=dir.name,
                database=database,
                like_file=f"{location}/{first_file.name}",
            )

        con.drop_table("win", database="ibis_testing", force=True)
        con.create_table(
            "win",
            schema=ibis.schema(dict(g="string", x="int", y="int")),
            database="ibis_testing",
        )
        con.raw_sql(
            """
            INSERT INTO ibis_testing.win VALUES
                ('a', 0, 3),
                ('a', 1, 2),
                ('a', 2, 0),
                ('a', 3, 1),
                ('a', 4, 1)
            """
        )
        assert con.list_tables(database="ibis_testing")

    def postload(self, **kw):
        env = IbisTestEnv()
        self.connection = self.connect(database=env.test_data_db, **kw)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        env = IbisTestEnv()
        return ibis.impala.connect(host=env.impala_host, port=env.impala_port, **kw)

    def _get_renamed_table(self, tablename: str) -> ir.Table:
        t = self.connection.table(tablename)
        original_names = TEST_TABLES[tablename].names
        return t.rename(dict(zip(original_names, t.columns)))

    @property
    def batting(self) -> ir.Table:
        return self._get_renamed_table("batting")

    @property
    def awards_players(self) -> ir.Table:
        return self._get_renamed_table("awards_players")


class IbisTestEnv:
    def __init__(self):
        if options.impala is None:
            ibis.backends.impala.Backend.register_options()

    @property
    def impala_host(self):
        return os.environ.get("IBIS_TEST_IMPALA_HOST", "localhost")

    @property
    def impala_port(self):
        return int(os.environ.get("IBIS_TEST_IMPALA_PORT", "21050"))

    @property
    def tmp_dir(self):
        leaf = util.gen_name("impala_test_tmp_dir")
        return os.environ.get("IBIS_TEST_TMP_DIR", f"/tmp/{leaf}")

    @property
    def test_data_db(self):
        return os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")

    @property
    def test_data_dir(self):
        return os.environ.get("IBIS_TEST_DATA_DIR", "/user/hive/warehouse")


@pytest.fixture(scope="session")
def env():
    return IbisTestEnv()


@pytest.fixture(scope="session")
def tmp_dir(env):
    return env.tmp_dir


@pytest.fixture(scope="session")
def test_data_db(env):
    return env.test_data_db


@pytest.fixture(scope="session")
def test_data_dir(env):
    return env.test_data_dir


@pytest.fixture(scope="session")
def backend(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id)


@pytest.fixture(scope="module")
def con(backend):
    return backend.connection


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture
def temp_parquet_table_schema():
    return ibis.schema(dict(id="int32", name="string", files="int32"))


@pytest.fixture
def temp_parquet_table(con, temp_parquet_table_schema):
    name = util.guid()
    yield con.create_table(name, schema=temp_parquet_table_schema, format="parquet")
    con.drop_table(name)


@pytest.fixture
def temp_parquet_table2(con, temp_parquet_table_schema):
    name = util.guid()
    yield con.create_table(name, schema=temp_parquet_table_schema, format="parquet")
    con.drop_table(name)


@pytest.fixture(scope="session")
def mockcon():
    return MockBackend()


@pytest.fixture(scope="module")
def kudu_table(con, test_data_db):
    name = "kudu_backed_table"
    with contextlib.closing(
        con.raw_sql(
            f"""\
CREATE TABLE {test_data_db}.{name} (
  a STRING,
  PRIMARY KEY (a)
)
PARTITION BY HASH PARTITIONS 2
STORED AS KUDU
TBLPROPERTIES (
  'kudu.master_addresses' = 'kudu',
  'kudu.num_tablet_replicas' = '1'
)"""
        )
    ):
        pass
    yield con.table(name)
    con.drop_table(name, database=test_data_db)


def translate(expr):
    return ibis.to_sql(expr, dialect="impala")
