from __future__ import annotations

import pytest
import sqlglot as sg

import ibis
from ibis.backends.bigquery import Backend, _force_quote_table, client


@pytest.mark.parametrize(
    ["project", "dataset", "expected"],
    [
        ("my-project", "", ("my-project", "my-project", "")),
        (
            "my-project",
            "my_dataset",
            ("my-project", "my-project", "my_dataset"),
        ),
        (
            "billing-project",
            "data-project.my_dataset",
            ("data-project", "billing-project", "my_dataset"),
        ),
    ],
)
def test_parse_project_and_dataset(project, dataset, expected):
    got = client.parse_project_and_dataset(project, dataset)
    assert got == expected


def test_parse_project_and_dataset_raises_error():
    expected_message = "data-project.my_dataset.table is not a BigQuery dataset"
    with pytest.raises(ValueError, match=expected_message):
        client.parse_project_and_dataset("my-project", "data-project.my_dataset.table")


@pytest.mark.parametrize(
    "bq_path_str, expected",
    [
        ("ibis-gbq.ibis_gbq_testing.argle", "`ibis-gbq`.`ibis_gbq_testing`.`argle`"),
        (
            "ibis-gbq.ibis_gbq_testing.28argle",
            "`ibis-gbq`.`ibis_gbq_testing`.`28argle`",
        ),
        ("mytable-287a", "`mytable-287a`"),
        ("myproject.mydataset.my-table", "`myproject`.`mydataset`.`my-table`"),
        ("my-dataset.mytable", "`my-dataset`.`mytable`"),
        (
            "a-7b0a.dev_test_dataset.test_ibis5",
            "`a-7b0a`.`dev_test_dataset`.`test_ibis5`",
        ),
    ],
)
def test_force_quoting(bq_path_str, expected):
    table = sg.parse_one(bq_path_str, into=sg.exp.Table, read="bigquery")
    table = _force_quote_table(table)

    assert table.sql("bigquery") == expected


def test_to_polars_keeps_microsecond_timestamps(monkeypatch):
    import datetime

    pa = pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")

    value = datetime.datetime(2025, 1, 1, 12, 34, 56, 123456)
    arrow = pa.table(
        {
            "ts": pa.array(
                [value.replace(tzinfo=datetime.UTC)], pa.timestamp("us", "UTC")
            ),
            "naive": pa.array([value], pa.timestamp("us")),
        }
    )
    con = Backend()
    monkeypatch.setattr(con, "to_pyarrow", lambda *_, **__: arrow)

    t = ibis.table({"ts": "timestamp('UTC')", "naive": "timestamp"}, name="t")

    df = con.to_polars(t)
    assert df.schema == pl.Schema(
        {"ts": pl.Datetime("us", "UTC"), "naive": pl.Datetime("us")}
    )
    assert df["naive"][0] == value

    def one_column(expr, *_, **__):
        return pa.table({expr.as_table().columns[0]: arrow["naive"]})

    monkeypatch.setattr(con, "to_pyarrow", one_column)
    assert con.to_polars(t.naive).dtype == pl.Datetime("us")
    assert con.to_polars(t.naive.max()) == value
