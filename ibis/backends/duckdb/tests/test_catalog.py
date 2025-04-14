from __future__ import annotations

import duckdb
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as exc
from ibis.util import gen_name


@pytest.fixture
def external_duckdb_file(tmpdir):  # pragma: no cover
    ddb_path = str(tmpdir / "starwars.ddb")
    con = ibis.duckdb.connect(ddb_path)

    try:
        starwars_df = pd.DataFrame(
            {
                "name": ["Luke Skywalker", "C-3PO", "R2-D2"],
                "height": [172, 167, 96],
                "mass": [77.0, 75.0, 32.0],
            }
        )
        con.create_table("starwars", obj=starwars_df)
    finally:
        con.disconnect()

    return ddb_path, starwars_df


def test_read_write_external_catalog(con, external_duckdb_file, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    ddb_path, starwars_df = external_duckdb_file
    name = gen_name("ext")
    con.attach(ddb_path, name=name)

    # Read from catalog
    assert name in con.list_catalogs()
    assert "main" in con.list_databases(catalog=name)

    db = f"{name}.main"

    assert "starwars" in con.list_tables(database=db)
    assert "starwars" not in con.list_tables()

    starwars = con.table("starwars", database=db)
    tm.assert_frame_equal(starwars.to_pandas(), starwars_df)

    # Write to catalog
    t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])

    _ = con.create_table("t2", obj=t, database=db)

    assert "t2" in con.list_tables(database=db)
    assert "t2" not in con.list_tables()

    table = con.table("t2", database=db)

    tm.assert_frame_equal(t.to_pandas(), table.to_pandas())

    # Overwrite table in catalog

    t_overwrite = ibis.memtable([{"a": 8, "b": "bing"}, {"a": 9, "b": "bong"}])

    _ = con.create_table("t2", obj=t_overwrite, database=db, overwrite=True)

    assert "t2" in con.list_tables(database=db)
    assert "t2" not in con.list_tables()

    table = con.table("t2", database=db)

    tm.assert_frame_equal(t_overwrite.to_pandas(), table.to_pandas())


def test_raise_if_catalog_and_temp(con):
    with pytest.raises(exc.UnsupportedArgumentError):
        con.create_table("some_table", obj="hi", temp=True, database="ext.main")


def test_cant_drop_database_external_catalog(con, tmpdir):
    name = gen_name("foobar")
    path = str(tmpdir / "f{name}.ddb")
    with duckdb.connect(path):
        pass
    con.attach(path)
    with pytest.raises(exc.UnsupportedOperationError):
        con.drop_database("main", catalog=name)
