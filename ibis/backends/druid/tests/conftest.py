from __future__ import annotations

import asyncio
import json
import os
import re
from functools import partial
from itertools import chain, repeat
from pathlib import Path
from typing import Any

import pytest
from aiohttp import ClientSession

import ibis
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest, ServiceSpec

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


async def wait_for_ingest(session: ClientSession, *, datasource: str) -> bool:
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
        async with session.get(
            url, params={"forceMetadataRefresh": next(force_refresh)}
        ) as resp:
            resp.raise_for_status()
            js = await resp.json()

        # floating point comparison ¯\_(ツ)_/¯
        all_segments_loaded = js[datasource] == 100.0
        await asyncio.sleep(REQUEST_INTERVAL)
    return all_segments_loaded


async def run_query(session: ClientSession, query: str) -> None:
    """Run a data loading query."""
    async with session.post(
        DDL_URL, data=json.dumps(dict(query=query)), headers=DDL_HEADERS
    ) as resp:
        resp.raise_for_status()
        js = await resp.json()

    task_id = js["taskId"]
    url = f"{BASE_URL}/indexer/v1/task/{task_id}/status"

    all_data_queryable = False

    while not all_data_queryable:
        async with session.get(url) as resp:
            resp.raise_for_status()
            js = await resp.json()

        status_blob = js["status"]
        status_string = status_blob["status"]
        if status_string == "SUCCESS":
            match = re.search(r'^REPLACE INTO "(?P<datasource>\w+)"', query)
            all_data_queryable = await wait_for_ingest(
                session, datasource=match.groupdict()["datasource"]
            )
        elif status_string == "FAILED":
            raise DruidDataLoadError(status_blob["errorMsg"])
        await asyncio.sleep(REQUEST_INTERVAL)


class TestConf(ServiceBackendTest, RoundHalfToEven):
    # druid has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    native_bool = True
    supports_structs = False
    supports_json = False  # it does, but we haven't implemented it

    @classmethod
    def service_spec(cls, data_dir: Path):
        files = [data_dir.joinpath("functional_alltypes.parquet")]
        files.extend(
            data_dir.joinpath("parquet", name, f"{name}.parquet")
            for name in ("diamonds", "batting", "awards_players")
        )
        return ServiceSpec(
            name="druid-coordinator", data_volume="/opt/shared", files=files
        )

    @staticmethod
    def _load_data(data_dir: Path, script_dir: Path, **_: Any) -> None:
        """Load test data into a druid backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        # copy data into the volume mount
        queries = filter(
            None,
            map(
                str.strip,
                (script_dir / 'schema' / 'druid.sql').read_text().split(";"),
            ),
        )

        # gather executes immediately, but we need to wait for asyncio.run to
        # create the event loop
        async def load_data(queries):
            """Run data loading queries."""
            async with ClientSession() as session:
                await asyncio.gather(*map(partial(run_query, session), queries))

        asyncio.run(load_data(queries))

    @staticmethod
    def connect(_: Path):
        return ibis.connect(DRUID_URL)


@pytest.fixture(scope='session')
def con():
    return ibis.connect(DRUID_URL)
