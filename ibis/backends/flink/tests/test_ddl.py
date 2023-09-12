from __future__ import annotations

import pytest
from py4j.protocol import Py4JJavaError

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch


@pytest.fixture
def awards_players_schema():
    return sch.Schema(
        {
            "playerID": dt.string,
            "awardID": dt.string,
            "yearID": dt.int32,
            "lgID": dt.string,
            "tie": dt.string,
            "notes": dt.string,
        }
    )


@pytest.fixture
def awards_players_csv_connector_configs():
    return {
        "connector": "filesystem",
        "path": "ci/ibis-testing-data/csv/awards_players.csv",
        "format": "csv",
        "csv.ignore-parse-errors": "true",
    }


def test_list_tables(con):
    assert len(con.list_tables())
    assert con.list_tables(catalog="default_catalog", database="default_database")


def test_create_table_from_schema(
    con, awards_players_schema, temp_table, awards_players_csv_connector_configs
):
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=awards_players_csv_connector_configs,
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema


@pytest.mark.parametrize("temp", [True, False])
def test_create_table(
    con, awards_players_schema, temp_table, awards_players_csv_connector_configs, temp
):
    con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=awards_players_csv_connector_configs,
        temp=temp,
    )
    assert temp_table in con.list_tables()

    if temp:
        with pytest.raises(Py4JJavaError):
            con.drop_table(temp_table)

    con.drop_table(temp_table, temp=temp)

    assert temp_table not in con.list_tables()
