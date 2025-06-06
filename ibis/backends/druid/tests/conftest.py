from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, repeat
from typing import TYPE_CHECKING, Any

from requests import Session

import ibis
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import ibis.expr.types as ir

DRUID_URL = os.environ.get(
    "DRUID_URL", "druid://localhost:8082/druid/v2/sql?header=true"
)

# A different endpoint is required for DDL queries :(
BASE_URL = "http://localhost:8888/druid"

DDL_URL = f"{BASE_URL}/v2/sql/task/"
DDL_HEADERS = {"Content-Type": "application/json"}

REQUEST_INTERVAL = 0.5  # seconds


class DruidDataLoadError(Exception):
    pass


def wait_for_ingest(session: Session, *, datasource: str) -> bool:
    """Wait for datasources to be queryable."""
    # https://druid.apache.org/docs/latest/ingestion/faq.html#how-do-i-know-when-i-can-make-query-to-druid-after-submitting-batch-ingestion-task
    #
    # Steps 1 and 2: If we're in this function then the ingestion task has been
    # submitted (step 1) and the task is complete (step 2).
    #
    # Steps 3 and 4: Poll the segment loading by datasource API once with
    # forced metadata refresh (step 3) and then continue to do without forcing
    # until all segments for the data source are loaded (step 4)
    all_segments_loaded = False
    url = f"{BASE_URL}/coordinator/v1/datasources/{datasource}/loadstatus"
    force_refresh = chain(("true",), repeat("false"))
    while not all_segments_loaded:
        resp = session.get(url, params={"forceMetadataRefresh": next(force_refresh)})
        resp.raise_for_status()
        js = resp.json()

        # floating point comparison ¯\_(ツ)_/¯
        all_segments_loaded = js[datasource] == 100.0
        time.sleep(REQUEST_INTERVAL)
    return all_segments_loaded


def run_query(session: Session, query: str) -> None:
    """Run a data loading query."""
    resp = session.post(
        DDL_URL, data=json.dumps(dict(query=query)), headers=DDL_HEADERS
    )
    resp.raise_for_status()
    js = resp.json()

    task_id = js["taskId"]
    url = f"{BASE_URL}/indexer/v1/task/{task_id}/status"

    all_data_queryable = False

    while not all_data_queryable:
        resp = session.get(url)
        resp.raise_for_status()
        js = resp.json()

        status_blob = js["status"]
        status_string = status_blob["status"]
        if status_string == "SUCCESS":
            match = re.search(r'^REPLACE INTO "(?P<datasource>\w+)"', query)
            all_data_queryable = wait_for_ingest(
                session, datasource=match.groupdict()["datasource"]
            )
        elif status_string == "FAILED":
            raise DruidDataLoadError(status_blob["errorMsg"])
        time.sleep(REQUEST_INTERVAL)


class TestConf(ServiceBackendTest):
    # druid has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = "ms"
    supports_arrays = False
    native_bool = True
    supports_structs = False
    supports_json = False  # it does, but we haven't implemented it
    rounding_method = "half_to_even"
    service_name = "druid-middlemanager"
    deps = ("pydruid.db",)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = self.connection.table("functional_alltypes")
        return t.mutate(
            # The parquet loading for booleans appears to be broken in Druid, so
            # I'm using this as a workaround to make the data match what's on disk.
            bool_col=1 - t.id % 2,
            # timestamp_col is loaded as a long because druid's type system is
            # awful: it does 99% of the work of a proper timestamp type, but
            # encodes it as an integer. I've never seen or heard of any other
            # tool that calls itself a time series database or "good for
            # working with time series", that lacks a first-class timestamp
            # type.
            timestamp_col=t.timestamp_col.as_timestamp("ms"),
        )

    @property
    def test_files(self) -> Iterable[Path]:
        return [
            path
            for path in self.data_dir.joinpath("parquet").glob("*.parquet")
            if path.name != "functional_alltypes.parquet"
        ] + [self.data_dir.joinpath("csv", "functional_alltypes.csv")]

    def _load_data(self, **_: Any) -> None:
        """Load test data into a druid backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        # run queries concurrently using threads; lots of time is spent on IO
        # making requests to check whether data loading is complete
        with Session() as session, ThreadPoolExecutor() as executor:
            for fut in as_completed(
                executor.submit(run_query, session, query) for query in self.ddl_script
            ):
                fut.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.connect(DRUID_URL, **kw)
