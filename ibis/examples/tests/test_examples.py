import os
import platform

import pytest

import ibis.examples
import ibis.util

pytestmark = pytest.mark.examples

duckdb = pytest.importorskip("duckdb")

ignored = {"wowah_data_raw"}  # this file is large (~80M)


@pytest.mark.parametrize("example", sorted(set(dir(ibis.examples)) - ignored))
@pytest.mark.duckdb
@pytest.mark.backend
@pytest.mark.xfail(
    (
        platform.system() == "Linux"
        and any(key.startswith("NIX_") for key in os.environ)
        and os.environ.get("IN_NIX_SHELL") != "impure"
    ),
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
