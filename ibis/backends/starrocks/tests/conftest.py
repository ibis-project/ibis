from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Any

import pytest
import requests

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.util import guid

if TYPE_CHECKING:
    from pathlib import Path


STARROCKS_USER = os.environ.get("IBIS_TEST_STARROCKS_USER", "root")
STARROCKS_PASS = os.environ.get("IBIS_TEST_STARROCKS_PASSWORD", "")
STARROCKS_HOST = os.environ.get("IBIS_TEST_STARROCKS_HOST", "127.0.0.1")
STARROCKS_PORT = int(os.environ.get("IBIS_TEST_STARROCKS_PORT", "9030"))
STARROCKS_HTTP_PORT = int(os.environ.get("IBIS_TEST_STARROCKS_HTTP_PORT", "8030"))
IBIS_TEST_STARROCKS_DB = os.environ.get("IBIS_TEST_STARROCKS_DATABASE", "ibis_testing")


def _stream_load_csv(*, table: str, csv_path: Path, tmpdir: Path) -> None:
    load_path = tmpdir / f"{table}-{guid()}.csv"
    with (
        csv_path.open(newline="", encoding="utf-8") as infile,
        load_path.open("w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator="\n")
        for row in reader:
            writer.writerow([r"\N" if value == "" else value for value in row])

    url = (
        f"http://{STARROCKS_HOST}:{STARROCKS_HTTP_PORT}/api/"
        f"{IBIS_TEST_STARROCKS_DB}/{table}/_stream_load"
    )
    headers = {
        "label": f"ibis_{table}_{guid()}",
        "format": "csv",
        "column_separator": ",",
        "skip_header": "1",
        "enclose": '"',
        "escape": "\\",
        "strict_mode": "false",
    }
    response = requests.put(
        url,
        auth=(STARROCKS_USER, STARROCKS_PASS),
        data=load_path.read_bytes(),
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Status") != "Success":
        raise RuntimeError(f"StarRocks Stream Load failed: {payload}")


class TestConf(BackendTest):
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    native_bool = False
    supports_structs = False
    rounding_method = "half_to_even"
    service_name = "starrocks"
    deps = ("MySQLdb",)

    def _load_data(self, **kwargs: Any) -> None:
        super()._load_data(**kwargs)

        tmpdir = kwargs["tmpdir"].getbasetemp()
        for table in TEST_TABLES:
            _stream_load_csv(
                table=table,
                csv_path=self.data_dir / "csv" / f"{table}.csv",
                tmpdir=tmpdir,
            )

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.starrocks.connect(
            host=STARROCKS_HOST,
            user=STARROCKS_USER,
            password=STARROCKS_PASS,
            database=IBIS_TEST_STARROCKS_DB,
            port=STARROCKS_PORT,
            autocommit=True,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
