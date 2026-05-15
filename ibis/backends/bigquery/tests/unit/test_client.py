from __future__ import annotations

import pytest
import sqlglot as sg

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


def test_create_view_project_dataset_database_uses_parsed_database():
    backend = Backend.__new__(Backend)
    backend.billing_project = "billing-project"
    backend.data_project = "billing-project"
    backend.dataset = "default_dataset"

    captured = {}

    backend._run_pre_execute_hooks = lambda _: None
    backend.compile = lambda _: sg.select(1)
    backend.raw_sql = lambda sql: captured.setdefault("sql", sql)
    backend.table = lambda name, /, *, database=None: captured.setdefault(
        "table", (name, database)
    )

    backend.create_view(
        "my_view",
        object(),
        database="my-project.my_dataset",
        overwrite=True,
    )

    assert captured["table"] == ("my_view", ("my-project", "my_dataset"))
