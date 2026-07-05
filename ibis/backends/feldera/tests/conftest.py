"""Pytest configuration for the Feldera backend tests.

Feldera is an incremental SQL engine.  Unlike most Ibis backends, tables cannot
be created ad hoc at runtime — they must be declared in a pipeline's SQL
program, and only *materialized* tables/views are queryable via ad-hoc SELECT.

So the test harness builds a single pipeline whose SQL program declares all the
standard Ibis test tables (as materialized tables), pushes the parquet test
data into them via ``Pipeline.input_pandas``, and yields an Ibis connection to
that running pipeline.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from filelock import FileLock

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest

# Extra tables (besides the standard TEST_TABLES) that some backend tests need.
EXTRA_TABLES = {
    "array_types": None,
    "json_t": None,
    "struct": None,
    "win": None,
    "topk": None,
}

FELDERA_HOST = os.environ.get(
    "IBIS_TEST_FELDERA_HOST", os.environ.get("FELDERA_HOST", "http://localhost:8080")
)

# pandas dtype -> Feldera/Calcite SQL type
_PANDAS_TO_FELDERA = {
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "uint8": "INTEGER",
    "uint16": "BIGINT",
    "uint32": "BIGINT",
    "uint64": "BIGINT",
    "float16": "REAL",
    "float32": "REAL",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "object": "VARCHAR",
    "string": "VARCHAR",
}


def _pandas_dtype_to_feldera(dtype) -> str:
    kind = dtype.kind
    if kind in "biu":
        return _PANDAS_TO_FELDERA.get(str(dtype), "BIGINT")
    if kind == "f":
        return _PANDAS_TO_FELDERA.get(str(dtype), "DOUBLE")
    if kind == "b":
        return "BOOLEAN"
    if kind == "M":
        return "TIMESTAMP"
    if kind == "U" or str(dtype) == "string":
        return "VARCHAR"
    # Fall back to VARCHAR for object/unknown dtypes.
    return "VARCHAR"


def _build_pipeline_sql(table_dtypes: dict[str, dict[str, str]]) -> str:
    """Build a CREATE TABLE ... WITH ('materialized'='true') program."""
    stmts = []
    for name, cols in table_dtypes.items():
        col_defs = ", ".join(f'"{c}" {t}' for c, t in cols.items())
        stmts.append(
            f"CREATE TABLE \"{name}\" ({col_defs}) WITH ('materialized' = 'true');"
        )
    return "\n".join(stmts)


def _read_parquet_dtypes(path) -> dict[str, str]:
    df = pd.read_parquet(path)
    return {col: _pandas_dtype_to_feldera(dt) for col, dt in df.dtypes.items()}


def _bootstrap_empty_pipeline() -> tuple[str, Any, Any]:
    """Create a minimal Feldera pipeline for no-data connection tests."""
    from feldera import FelderaClient, PipelineBuilder

    client = FelderaClient(FELDERA_HOST)
    name = f"ibis-nodata-{uuid.uuid4().hex[:8]}"
    pipe = PipelineBuilder(client, name=name, sql="-- ibis empty pipeline").create(
        wait=True
    )
    pipe.start()
    return name, client, pipe


def _bootstrap_pipeline(data_dir: Path) -> tuple[str, Any, Any]:
    """Create a Feldera pipeline with the standard Ibis test tables loaded."""
    from feldera import FelderaClient, PipelineBuilder

    table_dtypes: dict[str, dict[str, str]] = {}
    table_data: dict[str, pd.DataFrame] = {}

    for table_name in TEST_TABLES:
        path = data_dir / "parquet" / f"{table_name}.parquet"
        if path.exists():
            table_dtypes[table_name] = _read_parquet_dtypes(path)
            table_data[table_name] = pd.read_parquet(path)

    for extra in EXTRA_TABLES:
        path = data_dir / "parquet" / f"{extra}.parquet"
        if path.exists():
            table_dtypes[extra] = _read_parquet_dtypes(path)
            table_data[extra] = pd.read_parquet(path)

    sql = _build_pipeline_sql(table_dtypes)
    client = FelderaClient(FELDERA_HOST)
    name = f"ibis-test-{uuid.uuid4().hex[:8]}"
    pipe = PipelineBuilder(client, name=name, sql=sql).create(wait=True)
    pipe.start()

    for table_name, df in table_data.items():
        df.columns = [str(c) for c in df.columns]
        pipe.input_pandas(table_name, df)

    return name, client, pipe


class TestConf(BackendTest):
    """Feldera test configuration.

    Creates a fresh pipeline per test session, declares all standard test
    tables as materialized, pushes the parquet data in, and yields an Ibis
    connection bound to that pipeline.
    """

    # Feldera materialized output is unordered without ORDER BY; sort before compare.
    force_sort = True
    check_dtype = True
    supports_arrays = False  # TODO: validate; Calcite supports ARRAY types.
    supports_structs = False  # TODO: validate.
    supports_json = False
    supports_map = False
    stateful = False
    supports_tpch = False
    supports_tpcds = False
    deps = ("feldera",)

    @staticmethod
    def connect(*, tmpdir, worker_id, client=None, pipeline=None, **kw: Any):  # noqa: ARG004
        if pipeline is None:
            pipeline, client, _ = _bootstrap_empty_pipeline()
        return ibis.feldera.connect(client=client, pipeline=pipeline)

    @classmethod
    def load_data(
        cls, data_dir: Path, tmpdir, worker_id: str, **kw: Any
    ) -> BackendTest:
        """Bootstrap the Feldera pipeline before opening the Ibis connection."""
        root_tmp_dir = tmpdir.getbasetemp()
        if worker_id != "master":
            root_tmp_dir = root_tmp_dir.parent

        fn = root_tmp_dir / cls.name()
        with FileLock(f"{fn}.lock"):
            cls.skip_if_missing_deps()

            pipeline_name, client, pipe = _bootstrap_pipeline(data_dir)
            inst = cls(
                data_dir=data_dir,
                tmpdir=tmpdir,
                worker_id=worker_id,
                client=client,
                pipeline=pipeline_name,
                **kw,
            )
            inst._pipe = pipe

            if inst.stateful:
                inst.stateful_load(fn, **kw)
            else:
                if inst.supports_tpch:
                    inst.load_tpch()
                if inst.supports_tpcds:
                    inst.load_tpcds()

            inst.postload(tmpdir=tmpdir, worker_id=worker_id, **kw)
            return inst

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop and delete the pipeline so CI loops don't accumulate pipelines.
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:
            import time

            with contextlib.suppress(Exception):
                pipe.stop(force=True)
            for _ in range(10):
                try:
                    pipe.delete()
                    break
                except Exception:
                    time.sleep(1)
        self.connection.disconnect()


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    """Session-scoped Ibis connection to a Feldera pipeline with test data."""
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
