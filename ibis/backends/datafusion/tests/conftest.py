from __future__ import annotations

from typing import Any

import pytest
import sqlglot as sg

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.data import array_types, topk, win


class TestConf(BackendTest):
    # check_names = False
    # returned_timestamp_unit = 'ns'
    supports_structs = False
    supports_json = False
    supports_arrays = True
    supports_tpch = True
    supports_tpcds = True
    stateful = False
    deps = ("datafusion",)
    # Query 1 seems to require a bit more room here
    tpc_absolute_tolerance = 0.11

    def _load_data(self, **_: Any) -> None:
        con = self.connection
        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            con.read_parquet(path, table_name=table_name)
        con.create_table("array_types", array_types)
        con.create_table("win", win)
        con.create_table("topk", topk)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.datafusion.connect(**kw)

    def _load_tpc(self, *, suite, scale_factor):
        con = self.connection
        schema = f"tpc{suite}"
        con.create_database(schema)
        for path in self.data_dir.joinpath(
            schema, f"sf={scale_factor}", "parquet"
        ).glob("*.parquet"):
            table_name = path.with_suffix("").name
            con.con.sql(
                # datafusion can't create an external table in a specific schema it seems
                # so hack around that by
                #
                # 1. creating an external table in the current schema
                # 2. create an internal table in the desired schema using a
                #    CTAS from the external table
                # 3. drop the external table
                f"CREATE EXTERNAL TABLE {table_name} STORED AS PARQUET LOCATION '{path}'"
            )

            con.con.sql(
                f"CREATE TABLE {schema}.{table_name} AS SELECT * FROM {table_name}"
            )
            con.con.sql(f"DROP TABLE {table_name}")

    def _transform_tpc_sql(self, parsed, *, suite, leaves):
        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table) and node.name in leaves:
                return node.__class__(
                    catalog=f"tpc{suite}",
                    **{k: v for k, v in node.args.items() if k != "catalog"},
                )
            return node

        return parsed.transform(add_catalog_and_schema)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()
