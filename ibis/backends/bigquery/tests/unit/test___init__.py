from __future__ import annotations

import concurrent.futures
from datetime import date

import google.api_core.exceptions
import pytest
from google.cloud.bigquery import QueryJobConfig
from google.cloud.bigquery.query import ScalarQueryParameter
from google.cloud.bigquery.table import TableReference

import ibis
from ibis.backends.bigquery import Backend, _merge_params_into_config


def test_session_creation_is_bounded(mocker):
    """Bound session creation at the request, job, and polling layers."""
    client = mocker.Mock(project="billing-project", default_query_job_config=None)
    storage_client = mocker.Mock()
    query = client.query.return_value
    query.destination = TableReference.from_string(
        "billing-project.anonymous_dataset.anonymous_table"
    )

    backend = Backend()
    backend.do_connect(client=client, storage_client=storage_client)

    backend._make_session()

    client.query.assert_called_once_with(
        "SELECT 1",
        job_config=mocker.ANY,
        project="billing-project",
        timeout=60.0,
    )
    query.result.assert_called_once_with(timeout=60.0)
    assert client.query.call_args.kwargs["job_config"].to_api_repr() == {
        "query": {"useQueryCache": False},
        "jobTimeoutMs": "60000",
    }


@pytest.mark.parametrize(
    ("cancel_error", "expected_error"),
    [
        pytest.param(
            google.api_core.exceptions.GoogleAPICallError("cancellation failed"),
            concurrent.futures.TimeoutError,
            id="api-error",
        ),
        pytest.param(
            RuntimeError("cancellation failed"),
            RuntimeError,
            id="unexpected-error",
        ),
    ],
)
def test_session_creation_timeout_cancels_job(mocker, cancel_error, expected_error):
    """Bound cancellation and suppress only expected API failures."""
    client = mocker.Mock(project="billing-project", default_query_job_config=None)
    storage_client = mocker.Mock()
    query = client.query.return_value
    query.result.side_effect = concurrent.futures.TimeoutError
    query.cancel.side_effect = cancel_error

    backend = Backend()
    backend.do_connect(client=client, storage_client=storage_client)

    with pytest.raises(expected_error):
        backend._make_session()

    query.cancel.assert_called_once()
    cancel_kwargs = query.cancel.call_args.kwargs
    assert (cancel_kwargs["timeout"], cancel_kwargs["retry"].timeout) == (5.0, 5.0)


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
