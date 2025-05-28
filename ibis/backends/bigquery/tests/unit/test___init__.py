from __future__ import annotations

from datetime import date

import pytest
from google.cloud.bigquery import QueryJobConfig
from google.cloud.bigquery.query import ScalarQueryParameter
from google.cloud.bigquery.table import TableReference

import ibis
from ibis.backends.bigquery import _merge_params_into_config


@pytest.mark.parametrize(
    "query_job_config, params, expected",
    [
        (None, None, []),
        (QueryJobConfig(), None, []),
        (None, {}, []),
        (QueryJobConfig(), {}, []),
        (
            QueryJobConfig(
                query_parameters=[
                    ScalarQueryParameter("param1", "INT64", 1),
                    ScalarQueryParameter("param2", "INT64", 2),
                ],
            ),
            None,
            [
                ScalarQueryParameter("param1", "INT64", 1),
                ScalarQueryParameter("param2", "INT64", 2),
            ],
        ),
        (
            None,
            {
                ibis.literal(0).name("param1"): 1,
                ibis.literal(0).name("param2"): 2,
            },
            [
                ScalarQueryParameter("param1", "INT64", 1),
                ScalarQueryParameter("param2", "INT64", 2),
            ],
        ),
        (
            QueryJobConfig(
                query_parameters=[
                    ScalarQueryParameter("param1", "INT64", 1),
                    ScalarQueryParameter("param2", "INT64", 2),
                ],
            ),
            {
                ibis.literal(0).name("param2"): 3,
                ibis.literal(0).name("param3"): 4,
            },
            [
                ScalarQueryParameter("param1", "INT64", 1),
                ScalarQueryParameter("param2", "INT64", 3),
                ScalarQueryParameter("param3", "INT64", 4),
            ],
        ),
        (
            QueryJobConfig(
                query_parameters=[
                    ScalarQueryParameter("config1", "BOOL", True),
                    ScalarQueryParameter("config2", "INT64", 1),
                    ScalarQueryParameter("config3", "FLOAT64", 2.3),
                    ScalarQueryParameter("config4", "STRING", "abc"),
                    ScalarQueryParameter("config5", "DATE", "2025-01-01"),
                ],
                # ensure this is preserved
                destination=TableReference.from_string(
                    "test_project.test_dataset.test_table",
                ),
            ),
            {
                ibis.literal(False).name("param1"): False,
                ibis.literal(0).name("param2"): 4,
                ibis.literal(0.0).name("param3"): 5.6,
                ibis.literal("").name("param4"): "def",
                ibis.literal(date.today()).name("param5"): date(2025, 1, 2),
            },
            [
                ScalarQueryParameter("config1", "BOOL", True),
                ScalarQueryParameter("config2", "INT64", 1),
                ScalarQueryParameter("config3", "FLOAT64", 2.3),
                ScalarQueryParameter("config4", "STRING", "abc"),
                ScalarQueryParameter("config5", "DATE", date(2025, 1, 1)),
                ScalarQueryParameter("param1", "BOOL", False),
                ScalarQueryParameter("param2", "INT64", 4),
                ScalarQueryParameter("param3", "FLOAT64", 5.6),
                ScalarQueryParameter("param4", "STRING", "def"),
                ScalarQueryParameter("param5", "DATE", date(2025, 1, 2)),
            ],
        ),
    ],
)
def test__merge_params_into_config(query_job_config, params, expected):
    # check the merge is correct
    result = _merge_params_into_config(query_job_config, params)
    assert result is not query_job_config
    assert result.query_parameters == expected

    # check all the other fields are preserved
    if query_job_config is not None:
        expected_repr = query_job_config.to_api_repr()
        result_repr = result.to_api_repr()

        if "queryParameters" in expected_repr["query"]:
            del expected_repr["query"]["queryParameters"]

        if "queryParameters" in result_repr["query"]:
            del result_repr["query"]["queryParameters"]

        assert result_repr == expected_repr
