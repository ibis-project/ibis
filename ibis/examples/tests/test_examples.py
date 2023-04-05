import os

import pytest

import ibis.examples
import ibis.util
from ibis.backends.conftest import LINUX, SANDBOXED

pytestmark = pytest.mark.examples

duckdb = pytest.importorskip("duckdb")

# large files
ignored = frozenset(
    (
        "imdb_name_basics",
        "imdb_title_akas",
        "imdb_title_basics",
        "imdb_title_crew",
        "imdb_title_episode",
        "imdb_title_principals",
        "imdb_title_ratings",
        "wowah_data_raw",
    )
    * (os.environ.get("CI") is None)
)


@pytest.mark.parametrize("example", sorted(frozenset(dir(ibis.examples)) - ignored))
@pytest.mark.duckdb
@pytest.mark.backend
@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=OSError,
)
def test_examples(example, tmp_path):
    ex = getattr(ibis.examples, example)

    assert example in repr(ex)

    # initiate an new connection for every test case for isolation
    con = ibis.duckdb.connect(extension_directory=str(tmp_path))
    ibis.set_backend(con)

    df = ex.fetch().limit(1).execute()
    assert len(df) == 1


def test_non_example():
    gobbledygook = f"{ibis.util.guid()}"
    with pytest.raises(AttributeError, match=gobbledygook):
        getattr(ibis.examples, gobbledygook)
