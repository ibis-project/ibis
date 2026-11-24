from __future__ import annotations

from unittest.mock import MagicMock

import google.cloud.bigquery as bq
import pytest
import sqlglot as sg

import ibis
from ibis.backends.bigquery import _force_quote_table, client


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


@pytest.mark.parametrize(
    ("database", "expected_catalog", "expected_db"),
    [
        ("my-proj.my_dataset", "my-proj", "my_dataset"),
        (("my-proj", "my_dataset"), "my-proj", "my_dataset"),
        ("my_dataset", None, "my_dataset"),
        (None, None, None),
    ],
)
def test_create_view_database_arg(mocker, database, expected_catalog, expected_db):
    client_mock = MagicMock(spec=bq.Client)
    client_mock.project = "billing-proj"
    storage_mock = MagicMock()

    con = ibis.bigquery.connect(
        client=client_mock, storage_client=storage_mock, dataset_id="default_ds"
    )
    con.compile = mocker.MagicMock(return_value="SELECT 1")
    con.raw_sql = mocker.MagicMock()
    con.table = mocker.MagicMock()

    obj = mocker.MagicMock()
    con.create_view("my_view", obj, database=database)

    con.table.assert_called_once_with(
        "my_view", database=(expected_catalog, expected_db)
    )


def test_create_view_with_dataset_project_table_resolution(mocker):
    client_mock = MagicMock(spec=bq.Client)
    client_mock.project = "billing-proj"
    storage_mock = MagicMock()

    mock_bq_table = MagicMock()
    mock_bq_table.time_partitioning = None
    col = MagicMock()
    col.name = "x"
    col.field_type = "INTEGER"
    col.mode = "NULLABLE"
    col.fields = []
    mock_bq_table.schema = [col]
    client_mock.get_table.return_value = mock_bq_table

    con = ibis.bigquery.connect(
        client=client_mock, storage_client=storage_mock, dataset_id="default_ds"
    )
    con.compile = mocker.MagicMock(return_value="SELECT 1")
    con.raw_sql = mocker.MagicMock()

    obj = mocker.MagicMock()
    expr = con.create_view("my_view", obj, database="data-proj.custom_ds")

    client_mock.get_table.assert_called_once()
    table_ref = client_mock.get_table.call_args[0][0]
    assert table_ref.project == "data-proj"
    assert table_ref.dataset_id == "custom_ds"
    assert table_ref.table_id == "my_view"

    assert expr.get_name() == "data-proj.custom_ds.my_view"
    assert expr.op().name == "my_view"
    assert expr.op().namespace.catalog == "data-proj"
    assert expr.op().namespace.database == "custom_ds"
    assert expr.schema().names == ("x",)
