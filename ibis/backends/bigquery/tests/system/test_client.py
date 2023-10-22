from __future__ import annotations

import collections
import datetime
import decimal
import itertools

import pandas as pd
import pandas.testing as tm
import pytest
import pytz

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.bigquery.client import bigquery_param
from ibis.util import gen_name


def test_column_execute(alltypes, df):
    col_name = "float_col"
    expr = alltypes[col_name]
    result = expr.execute()
    expected = df[col_name]
    tm.assert_series_equal(
        # Sort the values because BigQuery doesn't guarantee row order unless
        # there is an order-by clause in the query.
        result.sort_values().reset_index(drop=True),
        expected.sort_values().reset_index(drop=True),
    )


def test_list_tables(con):
    tables = con.list_tables(like="functional_alltypes")
    assert set(tables) == {"functional_alltypes", "functional_alltypes_parted"}


def test_current_database(con, dataset_id):
    with pytest.warns(FutureWarning, match="data project"):
        db = con.current_database
    assert db == dataset_id
    assert db == con.dataset_id
    assert con.list_tables(schema=db, like="alltypes") == con.list_tables(
        like="alltypes"
    )


def test_array_collect(struct_table):
    key = struct_table.array_of_structs_col[0]["string_field"]
    expr = struct_table.group_by(key=key).aggregate(
        foo=lambda t: t.array_of_structs_col[0]["int_field"].collect()
    )
    result = expr.execute()
    expected = struct_table.execute()
    expected = (
        expected.assign(
            key=expected.array_of_structs_col.apply(lambda x: x[0]["string_field"])
        )
        .groupby("key")
        .apply(
            lambda df: list(df.array_of_structs_col.apply(lambda x: x[0]["int_field"]))
        )
        .reset_index()
        .rename(columns={0: "foo"})
    )
    tm.assert_frame_equal(result, expected)


def test_count_distinct_with_filter(alltypes):
    expr = alltypes.string_col.nunique(where=alltypes.string_col.cast("int64") > 1)
    result = expr.execute()
    expected = alltypes.string_col.execute()
    expected = expected[expected.astype("int64") > 1].nunique()
    assert result == expected


def test_cast_string_to_date(alltypes, df):
    string_col = alltypes.date_string_col
    month, day, year = map(string_col.split("/").__getitem__, range(3))

    expr = "20" + ibis.literal("-").join([year, month, day])
    expr = expr.cast("date")

    result = (
        expr.execute()
        .astype("datetime64[ns]")
        .sort_values()
        .reset_index(drop=True)
        .rename("date_string_col")
    )
    expected = (
        pd.to_datetime(df.date_string_col, format="%m/%d/%y")
        .dt.normalize()
        .sort_values()
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


def test_cast_float_to_int(alltypes, df):
    result = (alltypes.float_col - 2.55).cast("int64").to_pandas().sort_values()
    expected = (df.float_col - 2.55).astype("int64").sort_values()
    tm.assert_series_equal(result, expected, check_names=False)


def test_has_partitions(alltypes, parted_alltypes, con):
    col = con.partition_column
    assert col not in alltypes.columns
    assert col in parted_alltypes.columns


def test_different_partition_col_name(monkeypatch, con):
    col = "FOO_BAR"
    monkeypatch.setattr(con, "partition_column", col)
    alltypes = con.table("functional_alltypes")
    parted_alltypes = con.table("functional_alltypes_parted")
    assert col not in alltypes.columns
    assert col in parted_alltypes.columns


def test_subquery_scalar_params(alltypes, monkeypatch, snapshot):
    monkeypatch.setattr(ops.ScalarParameter, "_counter", itertools.count())
    t = alltypes
    p = ibis.param("timestamp").name("my_param")
    expr = (
        t[["float_col", "timestamp_col", "int_col", "string_col"]][
            lambda t: t.timestamp_col < p
        ]
        .group_by("string_col")
        .aggregate(foo=lambda t: t.float_col.sum())
        .foo.count()
        .name("count")
    )
    result = expr.compile(params={p: "20140101"})
    snapshot.assert_match(result, "out.sql")


def test_repr_struct_of_array_of_struct():
    name = "foo"
    p = ibis.param("struct<x: array<struct<y: array<double>>>>").name(name)
    value = collections.OrderedDict(
        [("x", [collections.OrderedDict([("y", [1.0, 2.0, 3.0])])])]
    )
    result = bigquery_param(p.type(), value, name)
    expected = {
        "name": "foo",
        "parameterType": {
            "structTypes": [
                {
                    "name": "x",
                    "type": {
                        "arrayType": {
                            "structTypes": [
                                {
                                    "name": "y",
                                    "type": {
                                        "arrayType": {"type": "FLOAT64"},
                                        "type": "ARRAY",
                                    },
                                }
                            ],
                            "type": "STRUCT",
                        },
                        "type": "ARRAY",
                    },
                }
            ],
            "type": "STRUCT",
        },
        "parameterValue": {
            "structValues": {
                "x": {
                    "arrayValues": [
                        {
                            "structValues": {
                                "y": {
                                    "arrayValues": [
                                        {"value": 1.0},
                                        {"value": 2.0},
                                        {"value": 3.0},
                                    ]
                                }
                            }
                        }
                    ]
                }
            }
        },
    }
    assert result.to_api_repr() == expected


def test_raw_sql(con):
    assert con.raw_sql("SELECT 1").fetchall() == [(1,)]


def test_parted_column_rename(parted_alltypes):
    assert "PARTITIONTIME" in parted_alltypes.columns
    assert "_PARTITIONTIME" in parted_alltypes.op().table.schema.names


def test_scalar_param_partition_time(parted_alltypes):
    assert "PARTITIONTIME" in parted_alltypes.columns
    assert "PARTITIONTIME" in parted_alltypes.schema()
    param = ibis.param("timestamp").name("time_param")
    expr = parted_alltypes[param > parted_alltypes.PARTITIONTIME]
    df = expr.execute(params={param: "2017-01-01"})
    assert df.empty


@pytest.mark.parametrize("kind", ["date", "timestamp"])
def test_parted_column(con, kind):
    table_name = f"{kind}_column_parted"
    t = con.table(table_name)
    expected_column = f"my_{kind}_parted_col"
    assert t.columns == [expected_column, "string_col", "int_col"]


def test_cross_project_query(public, snapshot):
    table = public.table("posts_questions")
    expr = table[table.tags.contains("ibis")][["title", "tags"]]
    result = expr.compile()
    snapshot.assert_match(result, "out.sql")
    n = 5
    df = expr.limit(n).execute()
    assert len(df) == n
    assert list(df.columns) == ["title", "tags"]
    assert df.title.dtype == object
    assert df.tags.dtype == object


def test_set_database(con2):
    con2.set_database("bigquery-public-data.epa_historical_air_quality")
    tables = con2.list_tables()
    assert "co_daily_summary" in tables


def test_exists_table_different_project(con):
    name = "co_daily_summary"
    dataset = "bigquery-public-data.epa_historical_air_quality"

    assert name in con.list_tables(schema=dataset)
    assert "foobar" not in con.list_tables(schema=dataset)


def test_multiple_project_queries(con, snapshot):
    so = con.table("posts_questions", database="bigquery-public-data.stackoverflow")
    trips = con.table("trips", database="nyc-tlc.yellow")
    join = so.join(trips, so.tags == trips.rate_code)[[so.title]]
    result = join.compile()
    snapshot.assert_match(result, "out.sql")


def test_multiple_project_queries_database_api(con, snapshot):
    stackoverflow = con.database("bigquery-public-data.stackoverflow")
    posts_questions = stackoverflow.posts_questions
    yellow = con.database("nyc-tlc.yellow")
    trips = yellow.trips
    predicate = posts_questions.tags == trips.rate_code
    join = posts_questions.join(trips, predicate)[[posts_questions.title]]
    result = join.compile()
    snapshot.assert_match(result, "out.sql")


def test_multiple_project_queries_execute(con):
    stackoverflow = con.database("bigquery-public-data.stackoverflow")
    posts_questions = stackoverflow.posts_questions.limit(5)
    yellow = con.database("nyc-tlc.yellow")
    trips = yellow.trips.limit(5)
    predicate = posts_questions.tags == trips.rate_code
    cols = [posts_questions.title]
    join = posts_questions.left_join(trips, predicate)[cols]
    result = join.execute()
    assert list(result.columns) == ["title"]
    assert len(result) == 5


def test_string_to_timestamp(con):
    timestamp = pd.Timestamp(
        datetime.datetime(year=2017, month=2, day=6), tz=pytz.timezone("UTC")
    )
    expr = ibis.literal("2017-02-06").to_timestamp("%F")
    result = con.execute(expr)
    assert result == timestamp

    timestamp_tz = pd.Timestamp(
        datetime.datetime(year=2017, month=2, day=6, hour=5),
        tz=pytz.timezone("UTC"),
    )
    expr_tz = ibis.literal("2017-02-06 America/New_York").to_timestamp("%F %Z")
    result_tz = con.execute(expr_tz)
    assert result_tz == timestamp_tz


def test_timestamp_column_parted_is_not_renamed(con):
    t = con.table("timestamp_column_parted")
    assert "_PARTITIONTIME" not in t.columns
    assert "PARTITIONTIME" not in t.columns


def test_numeric_table_schema(numeric_table):
    assert numeric_table.schema() == ibis.schema(
        [("string_col", dt.string), ("numeric_col", dt.Decimal(38, 9))]
    )


def test_numeric_sum(numeric_table):
    t = numeric_table
    expr = t.numeric_col.sum()
    result = expr.execute()
    assert isinstance(result, decimal.Decimal)
    compare = result.compare(decimal.Decimal("1.000000001"))
    assert compare == decimal.Decimal("0")


def test_boolean_casting(alltypes):
    t = alltypes
    expr = t.group_by(k=t.string_col.nullif("1") == "9").count()
    result = expr.execute().set_index("k")
    count = result.iloc[:, 0]
    assert count.at[False] == 5840
    assert count.at[True] == 730


def test_approx_median(alltypes):
    m = alltypes.month
    expected = m.execute().median()
    assert expected == 7

    expr = m.approx_median()
    result = expr.execute()
    # Since 6 and 7 are right on the edge for median in the range of months
    # (1-12), accept either for the approximate function.
    assert result in (6, 7)


def test_create_table_bignumeric(con, temp_table):
    schema = ibis.schema({"col1": dt.Decimal(76, 38)})
    temporary_table = con.create_table(temp_table, schema=schema)
    con.raw_sql(f"INSERT {con.current_schema}.{temp_table} (col1) VALUES (10.2)")
    df = temporary_table.execute()
    assert df.shape == (1, 1)


def test_geography_table(con, temp_table):
    schema = ibis.schema({"col1": dt.GeoSpatial(geotype="geography", srid=4326)})
    temporary_table = con.create_table(temp_table, schema=schema)
    con.raw_sql(
        f"INSERT {con.current_schema}.{temp_table} (col1) VALUES (ST_GEOGPOINT(1,3))"
    )
    df = temporary_table.execute()
    assert df.shape == (1, 1)

    assert temporary_table.schema() == ibis.schema(
        [("col1", dt.GeoSpatial(geotype="geography", srid=4326))]
    )


def test_timestamp_table(con, temp_table):
    schema = ibis.schema(
        {"datetime_col": dt.Timestamp(), "timestamp_col": dt.Timestamp(timezone="UTC")}
    )
    temporary_table = con.create_table(temp_table, schema=schema)
    con.raw_sql(
        f"INSERT {con.current_schema}.{temp_table} (datetime_col, timestamp_col) VALUES (CURRENT_DATETIME(), CURRENT_TIMESTAMP())"
    )
    df = temporary_table.execute()
    assert df.shape == (1, 2)

    assert temporary_table.schema() == ibis.schema(
        [
            ("datetime_col", dt.Timestamp()),
            ("timestamp_col", dt.Timestamp(timezone="UTC")),
        ]
    )


def test_fully_qualified_table_creation(con, project_id, dataset_id, temp_table):
    schema = ibis.schema({"col1": dt.GeoSpatial(geotype="geography", srid=4326)})
    t = con.create_table(f"{project_id}.{dataset_id}.{temp_table}", schema=schema)
    assert t.get_name() == f"{project_id}.{dataset_id}.{temp_table}"


def test_create_table_with_options(con):
    name = gen_name("bigquery_temp_table")
    schema = ibis.schema(dict(a="int64", b="int64", c="array<string>", d="date"))
    t = con.create_table(
        name,
        schema=schema,
        overwrite=True,
        default_collate="und:ci",
        partition_by="d",
        cluster_by=["a", "b"],
        options={
            "friendly_name": "bigquery_temp_table",
            "description": "A table for testing BigQuery's create_table implementation",
            "labels": [("org", "ibis")],
        },
    )
    try:
        assert t.execute().empty
    finally:
        con.drop_table(name)
