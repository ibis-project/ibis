from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import itertools
import operator
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import toolz

import ibis
import ibis.expr.types as ir
from ibis import options, util
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.impala.compiler import ImpalaCompiler, ImpalaExprTranslator
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero, UnorderedComparator
from ibis.backends.tests.data import win
from ibis.tests.expr.mocks import MockBackend


class TestConf(UnorderedComparator, BackendTest, RoundAwayFromZero):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False
    supports_divide_by_zero = True
    returned_timestamp_unit = "s"
    supports_structs = False
    supports_json = False
    deps = "fsspec", "requests", "impala"

    def _load_data(self, **_: Any) -> None:
        """Load test data into an Impala backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        import fsspec

        fs = fsspec.filesystem("file")

        data_files = fs.find(self.data_dir / "impala")

        # without setting the pool size
        # connections are dropped from the urllib3
        # connection pool when the number of workers exceeds this value.
        # this doesn't appear to be configurable through fsspec
        URLLIB_DEFAULT_POOL_SIZE = 10

        env = IbisTestEnv()
        futures = []
        con = self.connection
        con.create_database(env.test_data_db, force=True)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(
                os.environ.get("IBIS_DATA_MAX_WORKERS", URLLIB_DEFAULT_POOL_SIZE)
            )
        ) as executor:
            hdfs = con.hdfs
            tasks = {
                # make the database
                executor.submit(impala_create_test_database, con, env),
                # build and upload UDFs
                *itertools.starmap(
                    executor.submit,
                    impala_build_and_upload_udfs(hdfs, env, fs=fs),
                ),
                # upload data files
                *(
                    executor.submit(
                        hdfs_make_dir_and_put_file,
                        hdfs,
                        data_file,
                        os.path.join(
                            env.test_data_dir,
                            os.path.relpath(data_file, self.data_dir),
                        ),
                    )
                    for data_file in data_files
                ),
            }

            for future in concurrent.futures.as_completed(tasks):
                future.result()

            # create tables and compute stats
            compute_stats = operator.methodcaller("compute_stats")
            futures.append(
                executor.submit(
                    toolz.compose(compute_stats, con.avro_file),
                    os.path.join(env.test_data_dir, "impala", "avro", "tpch", "region"),
                    avro_schema={
                        "type": "record",
                        "name": "a",
                        "fields": [
                            {"name": "R_REGIONKEY", "type": ["null", "int"]},
                            {"name": "R_NAME", "type": ["null", "string"]},
                            {"name": "R_COMMENT", "type": ["null", "string"]},
                        ],
                    },
                    name="tpch_region_avro",
                    database=env.test_data_db,
                    persist=True,
                )
            )

            futures.extend(
                executor.submit(
                    toolz.compose(compute_stats, con.parquet_file),
                    path,
                    name=os.path.basename(path),
                    database=env.test_data_db,
                    persist=True,
                    schema=TEST_TABLES.get(os.path.basename(path)),
                )
                for path in con.hdfs.ls(
                    os.path.join(env.test_data_dir, "impala", "parquet")
                )
            )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    def postload(self, **kw):
        env = IbisTestEnv()
        self.connection = self.connect(database=env.test_data_db, **kw)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        import fsspec

        env = IbisTestEnv()
        con = ibis.impala.connect(
            host=env.impala_host,
            port=env.impala_port,
            auth_mechanism=env.auth_mechanism,
            hdfs_client=fsspec.filesystem(
                env.hdfs_protocol,
                host=env.nn_host,
                port=env.hdfs_port,
                user=env.hdfs_user,
            ),
            **kw,
        )
        return con

    def _get_original_column_names(self, tablename: str) -> list[str]:
        return list(TEST_TABLES[tablename].names)

    def _get_renamed_table(self, tablename: str) -> ir.Table:
        t = self.connection.table(tablename)
        original_names = self._get_original_column_names(tablename)
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
    def tmp_db(self):
        options.impala.temp_db = tmp_db = os.environ.get(
            "IBIS_TEST_TMP_DB", "ibis_testing_tmp_db"
        )
        return tmp_db

    @property
    def tmp_dir(self):
        options.impala.temp_hdfs_path = tmp_dir = os.environ.get(
            "IBIS_TEST_TMP_HDFS_DIR", f"/tmp/__ibis_test_{util.guid()}"
        )
        return tmp_dir

    @property
    def test_data_db(self):
        return os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")

    @property
    def test_data_dir(self):
        return os.environ.get("IBIS_TEST_DATA_HDFS_DIR", "/__ibis/ibis-testing-data")

    @property
    def nn_host(self):
        return os.environ.get("IBIS_TEST_NN_HOST", "localhost")

    @property
    def hdfs_port(self):
        return int(os.environ.get("IBIS_TEST_HDFS_PORT", 50070))

    @property
    def hdfs_superuser(self):
        return os.environ.get("IBIS_TEST_HDFS_SUPERUSER", "hdfs")

    @property
    def use_codegen(self):
        return ast.literal_eval(
            os.environ.get("IBIS_TEST_USE_CODEGEN", "False").lower().capitalize()
        )

    @property
    def auth_mechanism(self):
        return os.environ.get("IBIS_TEST_AUTH_MECH", "NOSASL")

    @property
    def hdfs_user(self):
        return os.environ.get("IBIS_TEST_HDFS_USER", "hdfs")

    @property
    def hdfs_protocol(self):
        return os.environ.get("IBIS_TEST_HDFS_PROTOCOL", "webhdfs")


@pytest.fixture(scope="session")
def env():
    return IbisTestEnv()


@pytest.fixture(scope="session")
def tmp_dir(env):
    options.impala.temp_hdfs_path = tmp_dir = env.tmp_dir
    return tmp_dir


@pytest.fixture(scope="session")
def test_data_db(env):
    return env.test_data_db


@pytest.fixture(scope="session")
def test_data_dir(env):
    return env.test_data_dir


@pytest.fixture(scope="session")
def hdfs(env, tmp_dir):
    fsspec = pytest.importorskip("fsspec")
    client = fsspec.filesystem(
        env.hdfs_protocol,
        host=env.nn_host,
        port=env.hdfs_port,
        user=env.hdfs_user,
    )

    if not client.exists(tmp_dir):
        client.mkdir(tmp_dir)
    return client


@pytest.fixture(scope="session")
def backend(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id)


@pytest.fixture(scope="module")
def con(env, backend):
    con = backend.connection
    con.disable_codegen(disabled=not env.use_codegen)
    assert con.get_options()["DISABLE_CODEGEN"] == str(int(not env.use_codegen))
    yield con
    con.close()


@pytest.fixture
def tmp_db(env, con):
    import impala

    tmp_db = env.tmp_db

    if tmp_db not in con.list_databases():
        con.create_database(tmp_db)
    yield tmp_db
    with contextlib.suppress(impala.error.HiveServer2Error):
        # The database can be dropped by another process during tear down
        # in the middle of dropping this one if tests are running in
        # parallel.
        #
        # We only care that it gets dropped before all tests are finished
        # running.
        con.drop_database(tmp_db, force=True)


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture
def temp_database(con):
    name = util.gen_name("database")
    con.create_database(name)
    yield name
    con.drop_database(name, force=True)


@pytest.fixture
def temp_table_db(con, temp_database):
    name = util.gen_name("table")
    yield temp_database, name
    assert name in con.list_tables(database=temp_database), name
    con.drop_table(name, database=temp_database)


@pytest.fixture
def temp_parquet_table_schema():
    return ibis.schema([("id", "int32"), ("name", "string"), ("files", "int32")])


@pytest.fixture
def temp_parquet_table(con, tmp_db, temp_parquet_table_schema):
    name = util.guid()
    db = con.database(tmp_db)
    db.create_table(name, schema=temp_parquet_table_schema, format="parquet")
    yield db[name]
    db.client.drop_table(name, database=tmp_db)


@pytest.fixture
def temp_parquet_table2(con, tmp_db, temp_parquet_table_schema):
    name = util.guid()
    db = con.database(tmp_db)
    db.create_table(name, schema=temp_parquet_table_schema, format="parquet")
    yield db[name]
    db.client.drop_table(name, database=tmp_db)


@pytest.fixture(scope="session")
def mockcon():
    return MockBackend()


@pytest.fixture(scope="module")
def kudu_table(con, test_data_db):
    name = "kudu_backed_table"
    cur = con.raw_sql(
        f"""\
CREATE TABLE {test_data_db}.{name} (
  a STRING,
  PRIMARY KEY(a)
)
PARTITION BY HASH PARTITIONS 2
STORED AS KUDU
TBLPROPERTIES (
  'kudu.master_addresses' = 'kudu',
  'kudu.num_tablet_replicas' = '1'
)"""
    )
    cur.close()
    yield con.table(name)
    con.drop_table(name, database=test_data_db)


def translate(expr, context=None, named=False):
    if context is None:
        context = ImpalaCompiler.make_context()
    translator = ImpalaExprTranslator(expr.op(), context=context, named=named)
    return translator.get_result()


def impala_build_and_upload_udfs(hdfs, env, *, fs):
    IBIS_HOME = Path(__file__).absolute().parents[4]
    cwd = str(IBIS_HOME / "ci" / "udf")
    subprocess.run(["cmake", ".", "-G", "Ninja"], cwd=cwd, check=True)
    subprocess.run(["ninja"], cwd=cwd, check=True)
    build_dir = IBIS_HOME / "ci" / "udf" / "build"
    bitcode_dir = os.path.join(env.test_data_dir, "udf")

    hdfs.mkdir(bitcode_dir, create_parents=True)

    for file in fs.find(build_dir):
        bitcode_path = os.path.join(bitcode_dir, os.path.relpath(file, build_dir))
        yield hdfs.put_file, file, bitcode_path


def hdfs_make_dir_and_put_file(fs, src, target):
    fs.mkdir(os.path.dirname(target), create_parents=True)
    fs.put_file(src, target)


def impala_create_test_database(con, env):
    con.drop_database(env.test_data_db, force=True)
    con.create_database(env.test_data_db)
    con.create_table(
        "alltypes",
        schema=ibis.schema(
            dict(
                a="int8",
                b="int16",
                c="int32",
                d="int64",
                e="float",
                f="double",
                g="string",
                h="boolean",
                i="timestamp",
            )
        ),
        database=env.test_data_db,
    )
    con.create_table(
        "win",
        schema=ibis.schema(dict(g="string", x="int64", y="int64")),
        database=env.test_data_db,
    )
    con.table("win", database=env.test_data_db).insert(win, overwrite=True)
