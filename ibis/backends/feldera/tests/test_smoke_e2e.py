"""End-to-end smoke test for the Ibis Feldera backend against a live Feldera.

Run::

    docker compose up feldera -d
    uv run pytest ibis/backends/feldera/tests/test_smoke_e2e.py -p no:cacheprovider -v
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from contextlib import contextmanager

import pandas as pd
import pytest

import ibis

HOST = os.environ.get(
    "FELDERA_HOST", os.environ.get("IBIS_TEST_FELDERA_HOST", "http://localhost:8080")
)

pytestmark = pytest.mark.feldera

SQL = """
CREATE TABLE penguins (
    species VARCHAR NOT NULL,
    island VARCHAR NOT NULL,
    bill_length_mm DOUBLE,
    bill_depth_mm DOUBLE,
    flipper_length_mm INTEGER,
    body_mass_g INTEGER,
    sex VARCHAR,
    year INTEGER
) WITH ('materialized' = 'true');

CREATE MATERIALIZED VIEW penguin_counts AS
SELECT species, island, COUNT(*) AS count
FROM penguins
GROUP BY species, island;
"""

ROWS = [
    {
        "species": "Adelie",
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "sex": "male",
        "year": 2007,
    },
    {
        "species": "Adelie",
        "island": "Torgersen",
        "bill_length_mm": 39.5,
        "bill_depth_mm": 17.4,
        "flipper_length_mm": 186,
        "body_mass_g": 3800,
        "sex": "female",
        "year": 2007,
    },
    {
        "species": "Gentoo",
        "island": "Biscoe",
        "bill_length_mm": 46.1,
        "bill_depth_mm": 13.2,
        "flipper_length_mm": 211,
        "body_mass_g": 4500,
        "sex": "female",
        "year": 2007,
    },
]


def _wait_for_ingest(
    pipe, table: str, expected_rows: int, timeout: float = 30.0
) -> None:
    """Poll until the table contains at least ``expected_rows`` rows."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batches = list(pipe.query_arrow(f"SELECT COUNT(*) FROM {table}"))
        if batches:
            count = next(iter(batches[0].to_pydict().values()))[0]
            if count >= expected_rows:
                return
        time.sleep(0.5)
    raise TimeoutError(
        f"Table {table!r} did not reach {expected_rows} rows within {timeout}s"
    )


@contextmanager
def _pipeline():
    from feldera import FelderaClient, PipelineBuilder

    client = FelderaClient(HOST)
    name = "ibis-e2e-" + uuid.uuid4().hex[:8]
    pipe = PipelineBuilder(client, name=name, sql=SQL).create(wait=True)
    try:
        pipe.start()
        pipe.input_pandas("penguins", pd.DataFrame(ROWS))
        _wait_for_ingest(pipe, "penguins", len(ROWS))
        yield name
    finally:
        with contextlib.suppress(Exception):
            pipe.stop(force=True)
        for _ in range(10):
            try:
                pipe.delete()
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)


def test_connect_and_metadata():
    with _pipeline() as name:
        con = ibis.feldera.connect(host=HOST, pipeline=name)
        version = con.version()
        assert version != "unknown"
        assert version.startswith("0.")
        tables = con.list_tables()
        assert "penguins" in tables
        assert "penguin_counts" in tables

        sch = con.get_schema("penguin_counts")
        assert set(sch.names) == {"species", "island", "count"}


def test_table_and_execute():
    with _pipeline() as name:
        con = ibis.feldera.connect(host=HOST, pipeline=name)
        t = con.table("penguin_counts")
        df = t.execute()
        # two groups: Adelie/Torgersen (count=2) and Gentoo/Biscoe (count=1)
        assert len(df) == 2
        counts = dict(zip(df["species"], df["count"]))
        assert counts["Adelie"] == 2
        assert counts["Gentoo"] == 1


def test_to_pyarrow():
    with _pipeline() as name:
        con = ibis.feldera.connect(host=HOST, pipeline=name)
        t = con.table("penguins")
        tbl = con.to_pyarrow(t)
        assert tbl.num_rows == 3
        assert "species" in tbl.column_names


def test_compile_only():
    """compile() must not require a live server (lazy client).

    ``con.table()`` calls ``get_schema()`` which does need a server, so build
    the table expression directly with an explicit schema.
    """
    con = ibis.feldera.connect(host="http://nonexistent:9999", pipeline="dummy")
    schema = ibis.schema(
        {"body_mass_g": "int64", "bill_length_mm": "float64", "species": "string"}
    )
    t = ibis.table(schema, name="penguins")
    expr = t.filter(t.body_mass_g > 3000).order_by(ibis.desc("body_mass_g"))
    sql = con.compile(expr)
    assert "ORDER BY" in sql
    assert "body_mass_g" in sql


def test_complex_expression():
    with _pipeline() as name:
        con = ibis.feldera.connect(host=HOST, pipeline=name)
        penguins = con.table("penguins")
        filtered = penguins.filter(penguins.body_mass_g > 3000)
        expr = (
            filtered.group_by("species")
            .agg(avg_bill=filtered.bill_length_mm.mean(), n=filtered.count())
            .order_by(ibis.desc("n"))
        )
        df = expr.execute()
        assert len(df) == 2
        # Adelie has 2 rows, Gentoo has 1, so Adelie should be first
        assert df.iloc[0]["species"] == "Adelie"
        assert df.iloc[0]["n"] == 2
