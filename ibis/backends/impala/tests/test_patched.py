from __future__ import annotations

import pandas as pd
import pytest

metadata = pytest.importorskip("ibis.backends.impala.metadata")


@pytest.fixture
def spy(con, mocker):
    return mocker.spy(con, "raw_sql")


@pytest.fixture
def qname(test_data_db):
    return f"`{test_data_db}`.`functional_alltypes`"


def test_invalidate_metadata(con, spy, test_data_db, qname):
    con.invalidate_metadata()
    spy.assert_called_with("INVALIDATE METADATA")

    con.invalidate_metadata("functional_alltypes")
    t = con.table("functional_alltypes")
    t.invalidate_metadata()

    con.invalidate_metadata("functional_alltypes", database=test_data_db)
    spy.assert_called_with(f"INVALIDATE METADATA {qname}")


def test_refresh(con, spy, qname):
    tname = "functional_alltypes"
    con.refresh(tname)
    spy.assert_called_with(f"REFRESH {qname}")

    t = con.table(tname)
    t.refresh()
    spy.assert_called_with(f"REFRESH {qname}")


def test_describe_formatted(con, spy, qname):
    t = con.table("functional_alltypes")
    desc = t.describe_formatted()
    spy.assert_called_with(f"DESCRIBE FORMATTED {qname}")
    assert isinstance(desc, metadata.TableMetadata)


def test_show_files(con, spy, qname):
    t = con.table("functional_alltypes")
    desc = t.files()
    spy.assert_called_with(f"SHOW FILES IN {qname}")
    assert isinstance(desc, pd.DataFrame)


def test_table_column_stats(con, spy, qname):
    t = con.table("functional_alltypes")

    desc = t.stats()
    spy.assert_called_with(f"SHOW TABLE STATS {qname}")
    assert isinstance(desc, pd.DataFrame)

    desc = t.column_stats()
    spy.assert_called_with(f"SHOW COLUMN STATS {qname}")
    assert isinstance(desc, pd.DataFrame)
