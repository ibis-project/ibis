from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as exc


@pytest.fixture(scope="session")
def external_duckdb_file(tmpdir_factory):  # pragma: no cover
    ddb_path = str(tmpdir_factory.mktemp("data") / "starwars.ddb")
    con = ibis.duckdb.connect(ddb_path)

    starwars_df = pd.DataFrame(
        {
            "name": ["Luke Skywalker", "C-3PO", "R2-D2"],
            "height": [172, 167, 96],
            "mass": [77.0, 75.0, 32.0],
        }
    )
    con.create_table("starwars", obj=starwars_df)
    con.disconnect()

    return ddb_path, starwars_df


def test_read_write_external_catalog(con, external_duckdb_file, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    ddb_path, starwars_df = external_duckdb_file
    con.attach(ddb_path, name="ext")

    # Read from catalog
    assert "ext" in con.list_catalogs()
    assert "main" in con.list_databases(catalog="ext")

    assert "starwars" in con.list_tables(database="ext.main")
    assert "starwars" not in con.list_tables()

    starwars = con.table("starwars", database="ext.main")
    tm.assert_frame_equal(starwars.to_pandas(), starwars_df)

    # Write to catalog
    t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])

    _ = con.create_table("t2", obj=t, database="ext.main")

    assert "t2" in con.list_tables(database="ext.main")
    assert "t2" not in con.list_tables()

    table = con.table("t2", database="ext.main")

    tm.assert_frame_equal(t.to_pandas(), table.to_pandas())

    # Overwrite table in catalog

    t_overwrite = ibis.memtable([{"a": 8, "b": "bing"}, {"a": 9, "b": "bong"}])

    _ = con.create_table("t2", obj=t_overwrite, database="ext.main", overwrite=True)

    assert "t2" in con.list_tables(database="ext.main")
    assert "t2" not in con.list_tables()

    table = con.table("t2", database="ext.main")

    tm.assert_frame_equal(t_overwrite.to_pandas(), table.to_pandas())


def test_raise_if_catalog_and_temp(con):
    with pytest.raises(exc.UnsupportedArgumentError):
        con.create_table("some_table", obj="hi", temp=True, database="ext.main")
