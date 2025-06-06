from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import sqlglot as sg

import ibis
from ibis.backends.tests.base import (
    ServiceBackendTest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any

EXASOL_USER = os.environ.get("IBIS_TEST_EXASOL_USER", "sys")
EXASOL_PASS = os.environ.get("IBIS_TEST_EXASOL_PASSWORD", "exasol")
EXASOL_HOST = os.environ.get("IBIS_TEST_EXASOL_HOST", "localhost")
EXASOL_PORT = int(os.environ.get("IBIS_TEST_EXASOL_PORT", "8563"))
IBIS_TEST_EXASOL_DB = os.environ.get("IBIS_TEST_EXASOL_DATABASE", "EXASOL")


class TestConf(ServiceBackendTest):
    check_dtype = False
    check_names = False
    supports_arrays = False
    returned_timestamp_unit = "us"
    native_bool = True
    supports_structs = False
    supports_json = False
    supports_map = False
    reduction_tolerance = 1e-7
    stateful = True
    service_name = "exasol"
    force_sort = True
    deps = ("pyexasol",)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):  # noqa: ARG004
        return ibis.exasol.connect(
            user=EXASOL_USER,
            password=EXASOL_PASS,
            host=EXASOL_HOST,
            port=EXASOL_PORT,
            **kw,
        )

    def postload(self, **kw: Any):
        self.connection = self.connect(schema=IBIS_TEST_EXASOL_DB, **kw)

    @staticmethod
    def format_table(name: str) -> str:
        return sg.to_identifier(name, quoted=True).sql("exasol")

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _exaplus(self) -> str:
        find_exaplus = [
            "docker",
            "compose",
            "exec",
            self.service_name,
            "find",
            "/opt",
            "-name",
            "exaplus",
            "-type",
            "f",  # only files
            "-executable",  # only executable files
            "-print",  # -print -quit will stop after the result is found
            "-quit",
        ]
        result = subprocess.run(
            find_exaplus, capture_output=True, check=True, text=True
        )
        return result.stdout.strip()

    def _load_data(self, **_: Any) -> None:
        """Load test data into a backend."""
        ddl_file = f"{self.data_volume}/exasol.sql"
        execute_ddl_file = [
            "docker",
            "compose",
            "exec",
            self.service_name,
            self._exaplus(),
            "-c",
            f"{EXASOL_HOST}:{EXASOL_PORT}",
            "-u",
            EXASOL_USER,
            "-p",
            EXASOL_PASS,
            "-f",
            ddl_file,
            "--jdbcparam",
            "validateservercertificate=0",
        ]
        subprocess.check_call(execute_ddl_file)

    def preload(self):
        # copy data files
        super().preload()

        service = self.service_name
        data_volume = self.data_volume
        path = self.script_dir / f"{self.name()}.sql"

        subprocess.check_call(
            [
                "docker",
                "compose",
                "cp",
                f"{path}",
                f"{service}:{data_volume}/{path.name}",
            ]
        )
