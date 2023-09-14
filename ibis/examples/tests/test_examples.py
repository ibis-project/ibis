from __future__ import annotations

import uuid

import pytest

import ibis.examples
import ibis.util
from ibis.conftest import CI, LINUX, MACOS, SANDBOXED

pytestmark = pytest.mark.examples

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pins")

# large files or files that are used elsewhere
ignored = frozenset(
    (
        # large
        "imdb_name_basics",
        "imdb_title_akas",
        "imdb_title_basics",
        "imdb_title_crew",
        "imdb_title_episode",
        "imdb_title_principals",
        "imdb_title_ratings",
        "wowah_data_raw",
    )
    * (not CI)  # ignore locally, but not in CI
    + (
        # use in doctests, avoid possible simultaneous use of the downloaded file
        "Aids2",
        "billboard",
        "fish_encounters",
        "penguins",
        "penguins_raw_raw",
        "relig_income_raw",
        "us_rent_income",
        "warpbreaks",
        "who",
        "world_bank_pop_raw",
    )
    * CI  # ignore in CI, but not locally
)

skip_linux_nix = pytest.mark.skipif(
    (LINUX or MACOS) and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)


@pytest.fixture(scope="session")
def example_con(tmp_path_factory, worker_id):
    return ibis.duckdb.connect(
        extension_directory=str(tmp_path_factory.mktemp(worker_id))
    )


@pytest.mark.parametrize("example", sorted(frozenset(dir(ibis.examples)) - ignored))
@pytest.mark.duckdb
@pytest.mark.backend
@skip_linux_nix
def test_examples(example, example_con):
    ex = getattr(ibis.examples, example)

    assert example in repr(ex)

    t = ex.fetch(backend=example_con)
    assert t.op().name == example

    n = t.limit(1).count().execute()
    assert n == 1


def test_non_example():
    gobbledygook = f"{ibis.util.guid()}"
    with pytest.raises(AttributeError, match=gobbledygook):
        getattr(ibis.examples, gobbledygook)


@pytest.mark.duckdb
@pytest.mark.backend
@skip_linux_nix
def test_backend_arg():
    con = ibis.duckdb.connect()
    t = ibis.examples.penguins.fetch(backend=con)
    assert t.get_name() in con.list_tables()


@pytest.mark.duckdb
@pytest.mark.backend
@skip_linux_nix
def test_table_name_arg():
    con = ibis.duckdb.connect()
    name = f"penguins-{uuid.uuid4().hex}"
    t = ibis.examples.penguins.fetch(backend=con, table_name=name)
    assert t.get_name() == name


@pytest.mark.pandas
@pytest.mark.duckdb
@pytest.mark.backend
@skip_linux_nix
@pytest.mark.parametrize(
    ("example", "columns"),
    [
        ("ml_latest_small_links", ["movieId", "imdbId", "tmdbId"]),
        ("band_instruments", ["name", "plays"]),
        (
            "AwardsManagers",
            ["player_id", "award_id", "year_id", "lg_id", "tie", "notes"],
        ),
    ],
    ids=["parquet", "csv", "csv-all-null"],
)
@pytest.mark.parametrize("backend_name", ["duckdb", "polars", "pandas"])
def test_load_example(backend_name, example, columns):
    pytest.importorskip(backend_name)
    con = getattr(ibis, backend_name).connect()
    t = getattr(ibis.examples, example).fetch(backend=con)
    assert t.columns == columns
