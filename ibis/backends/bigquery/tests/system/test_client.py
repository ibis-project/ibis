from __future__ import annotations

import collections
import datetime
import decimal
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from google.api_core.exceptions import Forbidden

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.bigquery.client import bigquery_param
from ibis.util import gen_name

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ibis.backends.bigquery import Backend


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

    pypi_tables = [
        "external",
        "native",
    ]

    assert con.list_tables()

    assert con.list_tables(database="ibis-gbq.pypi") == pypi_tables
    assert con.list_tables(database=("ibis-gbq", "pypi")) == pypi_tables


def test_current_catalog(con):
    assert con.current_catalog == con.billing_project


def test_current_database(con):
    assert con.current_database == con.dataset


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
    tm.assert_series_equal(result, expected, check_names=False, check_index=False)


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
    result = con.raw_sql("SELECT 1 as a").to_arrow()
    expected = pa.Table.from_pydict({"a": [1]})
    assert result.equals(expected)


def test_parted_column_rename(parted_alltypes):
    assert "PARTITIONTIME" in parted_alltypes.columns
    assert "_PARTITIONTIME" in parted_alltypes.op().parent.schema.names


def test_scalar_param_partition_time(parted_alltypes):
    assert "PARTITIONTIME" in parted_alltypes.columns
    assert "PARTITIONTIME" in parted_alltypes.schema()
    param = ibis.param("timestamp('UTC')")
    expr = parted_alltypes.filter(param > parted_alltypes.PARTITIONTIME)
    df = expr.execute(params={param: "2017-01-01"})
    assert df.empty


@pytest.mark.parametrize("kind", ["date", "timestamp"])
def test_parted_column(con, kind):
    table_name = f"{kind}_column_parted"
    t = con.table(table_name)
    expected_column = f"my_{kind}_parted_col"
    assert t.columns == (expected_column, "string_col", "int_col")


def test_cross_project_query(public):
    table = public.table("posts_questions")
    expr = table.filter(table.tags.contains("ibis"))[["title", "tags"]]
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

    assert name in con.list_tables(database=dataset)
    assert "foobar" not in con.list_tables(database=dataset)


@pytest.mark.xfail(
    condition=os.environ.get("GITHUB_ACTIONS") is not None,
    raises=Forbidden,
    reason="WIF auth not entirely worked out yet",
)
def test_multiple_project_queries_execute(con):
    posts_questions = con.table(
        "posts_questions", database="bigquery-public-data.stackoverflow"
    ).limit(5)
    trips = con.table("trips", database="nyc-tlc.yellow").limit(5)
    predicate = posts_questions.tags == trips.rate_code
    cols = [posts_questions.title]
    join = posts_questions.left_join(trips, predicate).select(cols)
    result = join.execute()
    assert list(result.columns) == ["title"]
    assert len(result) == 5


def test_string_as_timestamp(con):
    timestamp = pd.Timestamp(
        datetime.datetime(year=2017, month=2, day=6), tz=datetime.timezone.utc
    )
    expr = ibis.literal("2017-02-06").as_timestamp("%F")
    result = con.execute(expr)
    assert result == timestamp

    timestamp_tz = pd.Timestamp(
        datetime.datetime(year=2017, month=2, day=6, hour=5),
        tz=datetime.timezone.utc,
    )
    expr_tz = ibis.literal("2017-02-06 America/New_York").as_timestamp("%F %Z")
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
    assert compare == decimal.Decimal(0)


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
    con.raw_sql(f"INSERT {con.current_database}.{temp_table} (col1) VALUES (10.2)")
    df = temporary_table.execute()
    assert df.shape == (1, 1)


def test_geography_table(con, temp_table):
    pytest.importorskip("geopandas")

    schema = ibis.schema({"col1": dt.GeoSpatial(geotype="geography", srid=4326)})
    temporary_table = con.create_table(temp_table, schema=schema)
    con.raw_sql(
        f"INSERT {con.current_database}.{temp_table} (col1) VALUES (ST_GEOGPOINT(1,3))"
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
        f"INSERT {con.current_database}.{temp_table} (datetime_col, timestamp_col) VALUES (CURRENT_DATETIME(), CURRENT_TIMESTAMP())"
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


def test_fully_qualified_memtable_compile(project_id, dataset_id):
    new_bq_con = ibis.bigquery.connect(project_id=project_id, dataset_id=dataset_id)
    # New connection shouldn't have __session_dataset populated after
    # connection
    assert new_bq_con._Backend__session_dataset is None

    t = ibis.memtable(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        schema=ibis.schema({"a": "int64", "b": "int64"}),
    )

    # call to compile should fill in _session_dataset
    sql = new_bq_con.compile(t)
    assert new_bq_con._session_dataset is not None
    assert project_id in sql

    assert f"`{project_id}`.`{new_bq_con._session_dataset.dataset_id}`.`" in sql


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


def test_create_temp_table_from_scratch(project_id, dataset_id):
    con = ibis.bigquery.connect(project_id=project_id, dataset_id=dataset_id)
    name = gen_name("bigquery_temp_table")
    df = con.tables.functional_alltypes.limit(1)
    t = con.create_table(name, obj=df, temp=True)
    assert len(t.execute()) == 1


def test_create_table_from_scratch_with_spaces(project_id, dataset_id):
    con = ibis.bigquery.connect(project_id=project_id, dataset_id=dataset_id)
    name = f"{gen_name('bigquery_temp_table')} with spaces"
    df = con.tables.functional_alltypes.limit(1)
    t = con.create_table(name, obj=df)
    try:
        assert len(t.execute()) == 1
    finally:
        con.drop_table(name)


@pytest.mark.parametrize("ret_type", ["pandas", "pyarrow", "pyarrow_batches"])
def test_table_suffix(ret_type):
    con = ibis.connect("bigquery://ibis-gbq")
    t = con.table("gsod*", database="bigquery-public-data.noaa_gsod")
    expr = t.filter(t._TABLE_SUFFIX == "1929", t.max != 9999.9).head(1)
    if ret_type == "pandas":
        result = expr.to_pandas()
        cols = list(result.columns)
    elif ret_type == "pyarrow":
        result = expr.to_pyarrow()
        cols = result.column_names
    elif ret_type == "pyarrow_batches":
        result = pa.Table.from_batches(expr.to_pyarrow_batches())
        cols = result.column_names
    assert len(result)
    assert "_TABLE_PREFIX" not in cols


def test_parameters_in_url_connect(mocker):
    spy = mocker.spy(ibis.bigquery, "_from_url")
    parsed = urlparse("bigquery://ibis-gbq?location=us-east1")
    ibis.connect("bigquery://ibis-gbq?location=us-east1")
    spy.assert_called_once_with(parsed, location="us-east1")


def test_complex_column_name(con):
    expr = ibis.literal(1).name(
        "StringToTimestamp_StringConcat_date_string_col_' America_New_York'_'%F %Z'"
    )
    result = con.to_pandas(expr)
    assert result == 1


def test_geospatial_interactive(con, monkeypatch):
    pytest.importorskip("geopandas")

    monkeypatch.setattr(ibis.options, "interactive", True)
    t = con.table("bigquery-public-data.geo_us_boundaries.zip_codes")
    expr = (
        t.filter(lambda t: t.zip_code_geom.geometry_type() == "ST_Polygon")
        .head(1)
        .zip_code_geom
    )
    result = repr(expr)
    assert "POLYGON" in result


def test_geom_from_pyarrow(con, monkeypatch):
    shp = pytest.importorskip("shapely")

    monkeypatch.setattr(ibis.options, "interactive", True)

    data = pa.Table.from_pydict(
        {
            "id": [1, 2],
            "location": [
                shp.Point(1, 1).wkb,
                shp.Point(2, 2).wkb,
            ],
        }
    )

    # Create table in BigQuery
    name = gen_name("bq_test_geom")
    schema = ibis.schema({"id": "int64", "location": "geospatial:geography;4326"})

    t = con.create_table(name, data, schema=schema)

    try:
        assert repr(t)
        assert len(t.to_pyarrow()) == 2
        assert len(t.to_pandas()) == 2
    finally:
        con.drop_table(name)


def test_raw_sql_params_with_alias(con):
    name = "cutoff"
    cutoff = ibis.param("date").name(name)
    value = datetime.date(2024, 10, 28)
    query_parameters = {cutoff: value}
    result = con.raw_sql(f"SELECT @{name} AS {name}", params=query_parameters)
    assert list(map(dict, result)) == [{name: value}]


@pytest.fixture(scope="module")
def tmp_table(con):
    data = pd.DataFrame(
        {"foo": [1, 1, 2, 2, 3, 3], "bar": ["a", "b", "a", "a", "b", "b"]}
    )
    name = gen_name("test_window_with_count_distinct")
    test_table = con.create_table(name, data)
    yield test_table
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("expr", "query"),
    [
        (
            lambda t: t.group_by("foo").mutate(bar=lambda t: t.bar.nunique()),
            "SELECT foo, COUNT(DISTINCT bar) OVER (PARTITION BY foo) AS bar FROM {}".format,
        ),
        (
            lambda t: t.filter(
                lambda t: t.bar.nunique().over(ibis.window(group_by="foo")) > 1
            ),
            "SELECT * FROM {} QUALIFY COUNT(DISTINCT bar) OVER (PARTITION BY foo) > 1".format,
        ),
    ],
    ids=["project", "qualify"],
)
def test_window_with_count_distinct(tmp_table, expr, query):
    identifier = tmp_table.get_name()
    sql = query(identifier)
    result = (
        expr(tmp_table).to_pandas().sort_values(["foo", "bar"]).reset_index(drop=True)
    )
    expected = (
        tmp_table.sql(sql)
        .to_pandas()
        .sort_values(["foo", "bar"])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)


def test_query_with_job_id_prefix(con3: Backend):
    job_id_prefix = "ibis_test_"  # defined in con3 fixture
    query = "SELECT 1"
    result = con3.raw_sql(query)
    assert result.job_id.startswith(job_id_prefix)


def test_read_csv_with_custom_load_job_prefix(
    con3: Backend, mocker: MockerFixture, tmpdir
):
    """
    Since methods that upload data to BigQuery (like `read_csv`) don't return any data,
    they also don't return a job where we can inspect the job ID, so it's a little
    awkward to test that the job ID prefix is set correctly. This does it indirectly
    by spying on the `query` method of the client, which is called with the job ID
    prefix when the data is uploaded, and we trust that the BQ library uses it correctly.
    Else, this test tries to be flexible to allow internal changes in the implementation
    of the `read_csv` method.
    """
    job_id_prefix = "ibis_test_"  # defined in con3 fixture
    table_name = gen_name("test_table_with_custom_job_prefixes")

    path = tmpdir.join("test_data.csv")

    pd.DataFrame({"a": [1], "b": ["x"]}).to_csv(path, index=False)

    t = con3.read_csv(path, table_name=table_name)

    query_spy = mocker.spy(con3.client, "query")

    assert t.count().execute() > 0
    args, kwargs = query_spy.call_args
    kwargs = kwargs.copy()
    del kwargs["job_id_prefix"]
    query_spy.assert_called_once_with(*args, job_id_prefix=job_id_prefix, **kwargs)


def test_insert_with_custom_load_job_prefix(con3: Backend, mocker: MockerFixture):
    """
    Since methods that upload data to BigQuery (like `insert`) don't return any data,
    they also don't return a job where we can inspect the job ID, so it's a little
    awkward to test that the job ID prefix is set correctly. This does it indirectly
    by spying on the `query` method of the client, which is called with the job ID
    prefix when the data is uploaded, and we trust that the BQ library uses it correctly.
    Else, this test tries to be flexible to allow internal changes in the implementation
    of the `insert` method.
    """
    job_id_prefix = "ibis_test_"  # defined in con3 fixture

    df = pd.DataFrame({"a": [1], "b": ["x"]})
    table_name = gen_name("test_table_with_custom_job_prefixes")

    con3.create_table(table_name, schema={"a": "int", "b": "string"})
    con3.insert(table_name, obj=df)

    expr = con3.table(table_name).count()

    query_spy = mocker.spy(con3.client, "query")
    assert expr.execute()
    args, kwargs = query_spy.call_args
    kwargs = kwargs.copy()
    del kwargs["job_id_prefix"]
    query_spy.assert_called_once_with(*args, job_id_prefix=job_id_prefix, **kwargs)
