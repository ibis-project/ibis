"""Export paths not covered by the shared suite."""

from __future__ import annotations

from ibis import _


def test_to_pyarrow_batches_multiple_batches(mem):
    # chDB is notyet on the shared chunk-size tests, so this is the only
    # coverage that streaming yields more than one batch.
    n = 200_000
    t = mem.sql(f"SELECT number AS n FROM numbers({n})")
    reader = mem.to_pyarrow_batches(t, chunk_size=65_536)
    batches = list(reader)
    assert len(batches) > 1
    assert sum(b.num_rows for b in batches) == n


def test_empty_result_preserves_schema(mem):
    # an empty result from a real query returns zero columns, which the
    # converter must rebuild against the expected schema
    t = mem.sql("SELECT number AS n FROM numbers(10)").filter(_.n > 100)
    tbl = mem.to_pyarrow(t)
    assert tbl.num_rows == 0
    assert tbl.schema.names == ["n"]
