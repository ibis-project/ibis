import pytest

import ibis.examples
import ibis.util
from ibis.backends.conftest import LINUX, SANDBOXED

pytestmark = pytest.mark.examples

duckdb = pytest.importorskip("duckdb")

ignored = {"wowah_data_raw"}  # this file is large (~80M)


@pytest.mark.parametrize("example", sorted(set(dir(ibis.examples)) - ignored))
@pytest.mark.duckdb
@pytest.mark.backend
@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=OSError,
)
def test_examples(example):
    ex = getattr(ibis.examples, example)

    assert example in repr(ex)

    t = ex.fetch()
    df = t.limit(1).execute()
    assert len(df) == 1


def test_non_example():
    gobbledygook = f"{ibis.util.guid()}"
    with pytest.raises(AttributeError, match=gobbledygook):
        getattr(ibis.examples, gobbledygook)
