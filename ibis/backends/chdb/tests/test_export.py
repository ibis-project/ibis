"""Export paths not covered by the shared suite."""

from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
from ibis import _, util


def test_to_pyarrow_batches_multiple_batches(mem):
    n = 200_000
    t = mem.sql(f"SELECT number AS n FROM numbers({n})")
    reader = mem.to_pyarrow_batches(t, chunk_size=65_536)
    batches = list(reader)
    assert len(batches) > 1
    assert all(b.num_rows <= 65_536 for b in batches)
    assert sum(b.num_rows for b in batches) == n


def test_empty_result_preserves_schema(mem):
    # an empty result from a real query returns zero columns, which the
    # converter must rebuild against the expected schema
    t = mem.sql("SELECT number AS n FROM numbers(10)").filter(_.n > 100)
    tbl = mem.to_pyarrow(t)
    assert tbl.num_rows == 0
    assert tbl.schema.names == ["n"]


def test_interrupted_batch_reader_raises(mem):
    # another query ends the open stream; the reader must raise, not truncate
    t = mem.sql("SELECT number AS n FROM numbers(300000)")
    reader = mem.to_pyarrow_batches(t, chunk_size=65_536)
    reader.read_next_batch()
    mem.execute(ibis.literal(1))
    with pytest.raises(com.IbisError, match="interrupted"):
        util.consume(reader)


@pytest.mark.parametrize("shared", [False, True], ids=["same", "shared-session"])
@pytest.mark.parametrize("read", [False, True], ids=["drop", "read"])
def test_interrupted_reader_keeps_newer_stream(mem, shared, read):
    other = ibis.chdb.from_connection(mem.con._session) if shared else mem
    t = mem.sql("SELECT number AS n FROM numbers(300000)")
    stale = mem.to_pyarrow_batches(t, chunk_size=65_536)
    stale.read_next_batch()
    fresh = other.to_pyarrow_batches(t, chunk_size=65_536)
    if read:
        with pytest.raises(com.IbisError, match="interrupted"):
            util.consume(stale)
    del stale
    assert sum(b.num_rows for b in fresh) == 300_000
