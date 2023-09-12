from __future__ import annotations

import pytest
from py4j.protocol import Py4JJavaError

import ibis
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
def functiona_alltypes_schema():
    return sch.Schema(
        {
            "id": dt.int32,
            "bool_col": dt.bool,
            "smallint_col": dt.int16,
            "int_col": dt.int32,
            "bigint_col": dt.int64,
            "float_col": dt.float32,
            "double_col": dt.float64,
            "date_string_col": dt.string,
            "string_col": dt.string,
            "timestamp_col": dt.timestamp(scale=3),
            "year": dt.int32,
            "month": dt.int32,
        }
    )


@pytest.fixture
def csv_connector_configs():
    def generate_csv_configs(csv_file):
        return {
            "connector": "filesystem",
            "path": f"ci/ibis-testing-data/csv/{csv_file}.csv",
            "format": "csv",
            "csv.ignore-parse-errors": "true",
        }

    return generate_csv_configs


def test_list_tables(con):
    assert len(con.list_tables())
    assert con.list_tables(catalog="default_catalog", database="default_database")


def test_create_table_from_schema(
    con, awards_players_schema, temp_table, csv_connector_configs
):
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_connector_configs("awards_players"),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema


@pytest.mark.parametrize("temp", [True, False])
def test_create_table(
    con, awards_players_schema, temp_table, csv_connector_configs, temp
):
    con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_connector_configs("awards_players"),
        temp=temp,
    )
    assert temp_table in con.list_tables()

    if temp:
        with pytest.raises(Py4JJavaError):
            con.drop_table(temp_table)

    con.drop_table(temp_table, temp=temp)

    assert temp_table not in con.list_tables()


def test_create_source_table_with_watermark(
    con, functiona_alltypes_schema, temp_table, csv_connector_configs
):
    new_table = con.create_table(
        temp_table,
        schema=functiona_alltypes_schema,
        tbl_properties=csv_connector_configs("functional_alltypes"),
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == functiona_alltypes_schema
