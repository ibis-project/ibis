from __future__ import annotations

import concurrent.futures
import os
import tempfile
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve

import pyarrow.parquet as pq
import pyarrow_hotfix  # noqa: F401
import pytest
import snowflake.connector as sc
import sqlglot as sg

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.sql.datatypes import SnowflakeType
from ibis.backends.tests.base import BackendTest
from ibis.formats.pyarrow import PyArrowSchema

if TYPE_CHECKING:
    from ibis.backends import BaseBackend


def _get_url():
    if (url := os.environ.get("SNOWFLAKE_URL")) is not None:
        return url
    elif os.environ.get("SNOWFLAKE_HOME"):
        import tomli

        # requires a connection named ibis_testing or one explicitly set with
        # export SNOWFLAKE_DEFAULT_CONNECTION_NAME
        connection_name = os.environ.get(
            "SNOWFLAKE_DEFAULT_CONNECTION_NAME", "ibis_testing"
        )
        config_file = Path(os.environ["SNOWFLAKE_HOME"], "connections.toml")
        config_text = config_file.read_text()
        params = tomli.loads(config_text)[connection_name]
        user, password, account, database, schema, warehouse = (
            params["user"],
            params["password"],
            params["account"],
            params["database"],
            params["schema"],
            params["warehouse"],
        )
        return f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"
    else:
        try:
            user, password, account, database, schema, warehouse = tuple(
                os.environ[f"SNOWFLAKE_{part}"]
                for part in [
                    "USER",
                    "PASSWORD",
                    "ACCOUNT",
                    "DATABASE",
                    "SCHEMA",
                    "WAREHOUSE",
                ]
            )
        except KeyError as e:
            pytest.skip(
                f"missing URL part: {e} or SNOWFLAKE_URL environment variable is not defined"
            )
        return f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


def copy_into(con, data_dir: Path, table: str) -> None:
    stage = "ibis_testing"
    file = data_dir.joinpath("parquet", f"{table}.parquet").absolute()
    schema = PyArrowSchema.to_ibis(pq.read_metadata(file).schema.to_arrow_schema())
    columns = ", ".join(
        f"$1:{name}{'::VARCHAR' * typ.is_timestamp()}::{SnowflakeType.to_string(typ)}"
        for name, typ in schema.items()
    )
    con.execute(f"PUT {file.as_uri()} @{stage}/{file.name}")
    con.execute(
        f"""
        COPY INTO "{table}"
        FROM (SELECT {columns} FROM @{stage}/{file.name})
        FILE_FORMAT = (TYPE = PARQUET)
        """
    )


class TestConf(BackendTest):
    supports_map = True
    deps = ("snowflake.connector",)
    supports_tpch = True
    supports_tpcds = True

    def _load_tpc(self, *, suite, scale_factor) -> None:
        """Create views of data in the TPC-H catalog that ships with Trino.

        This method create relations that have column names prefixed with the
        first one (or two in the case of partsupp -> ps) character table name
        to match the DuckDB TPC-H query conventions.
        """
        con = self.connection
        schema = f"tpc{suite}"

        con.create_database(schema, force=True)

        parquet_dir = self.data_dir.joinpath(schema, f"sf={scale_factor}", "parquet")
        assert parquet_dir.exists(), parquet_dir

        tables = frozenset(con.list_tables(database=("IBIS_TESTING", schema)))
        for path in parquet_dir.glob("*.parquet"):
            table_name = path.with_suffix("").name
            if table_name not in tables:
                con.create_table(
                    table_name,
                    con.read_parquet(path),
                    database=f'IBIS_TESTING."{schema}"',
                )

    def _transform_tpc_sql(self, parsed, *, suite, leaves):
        def quote(node):
            if isinstance(node, sg.exp.Identifier):
                return sg.to_identifier(node.name, quoted=True)
            return node

        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table) and node.name in leaves:
                node.args["db"] = sg.to_identifier(f"tpc{suite}")
                node.args["catalog"] = sg.to_identifier("IBIS_TESTING")
            return node

        result = parsed.transform(add_catalog_and_schema)
        result = result.transform(quote)
        return result

    def _load_data(self, **_: Any) -> None:
        """Load test data into a Snowflake backend instance."""

        connect_args = {}

        url = urlparse(_get_url())

        if url.path:
            db, schema = url.path[1:].split("/", 1)
            (warehouse,) = parse_qs(url.query)["warehouse"]
            connect_args.update(
                {
                    "user": url.username,
                    "password": url.password,
                    "account": url.hostname,
                    "warehouse": warehouse,
                }
            )
        else:
            db = os.environ["SNOWFLAKE_DATABASE"]
            schema = os.environ["SNOWFLAKE_SCHEMA"]

        dbschema = f"{db}.{schema}"

        with closing(sc.connect(**connect_args)) as con, closing(con.cursor()) as c:
            c.execute("ALTER SESSION SET MULTI_STATEMENT_COUNT = 0 JSON_INDENT = 0")
            c.execute(
                f"""
                CREATE DATABASE IF NOT EXISTS {db};
                CREATE SCHEMA IF NOT EXISTS {dbschema};
                USE {dbschema};
                CREATE TEMP STAGE {db};
                CREATE STAGE IF NOT EXISTS ibis_testing;
                CREATE STAGE IF NOT EXISTS models;
                {self.script_dir.joinpath("snowflake.sql").read_text()}
                """
            )

            with tempfile.TemporaryDirectory() as d:
                path, _ = urlretrieve(
                    "https://storage.googleapis.com/ibis-testing-data/model.joblib",
                    os.path.join(d, "model.joblib"),
                )

                assert os.path.exists(path)
                assert os.path.getsize(path)

                c.execute(f"PUT {Path(path).as_uri()} @MODELS")

            # not much we can do to make this faster, but running these in
            # multiple threads seems to save about 2x
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for future in concurrent.futures.as_completed(
                    exe.submit(copy_into, c, self.data_dir, table)
                    for table in TEST_TABLES.keys()
                ):
                    future.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        if os.environ.get("SNOWFLAKE_SNOWPARK"):
            import snowflake.snowpark as sp

            if connection_name := os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME"):
                builder = sp.Session.builder.config("connection_name", connection_name)
            else:
                builder = sp.Session.builder.configs(
                    {
                        "user": os.environ["SNOWFLAKE_USER"],
                        "account": os.environ["SNOWFLAKE_ACCOUNT"],
                        "password": os.environ["SNOWFLAKE_PASSWORD"],
                        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
                        "database": os.environ["SNOWFLAKE_DATABASE"],
                        "schema": os.environ["SNOWFLAKE_SCHEMA"],
                    }
                )
            return ibis.backends.snowflake.Backend.from_connection(builder.create())
        else:
            return ibis.connect(_get_url(), **kw)


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
