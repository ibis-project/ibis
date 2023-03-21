import os
import platform
import tempfile
from pathlib import Path

import duckdb
import pytest

import ibis


def test_temp_directory(tmp_path):
    query = "SELECT current_setting('temp_directory')"

    # 1. in-memory + no temp_directory specified
    con = ibis.duckdb.connect()
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
        assert value  # we don't care what the specific value is

    temp_directory = Path(tempfile.gettempdir()) / "duckdb"

    # 2. in-memory + temp_directory specified
    con = ibis.duckdb.connect(temp_directory=temp_directory)
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
    assert value == str(temp_directory)

    # 3. on-disk + no temp_directory specified
    # untested, duckdb sets the temp_directory to something implementation
    # defined

    # 4. on-disk + temp_directory specified
    con = ibis.duckdb.connect(tmp_path / "test2.ddb", temp_directory=temp_directory)
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
    assert value == str(temp_directory)


def test_set_temp_dir(tmp_path):
    path = tmp_path / "foo" / "bar"
    ibis.duckdb.connect(temp_directory=path)
    assert path.exists()


@pytest.mark.xfail(
    (
        platform.system() == "Linux"
        and any(key.startswith("NIX_") for key in os.environ)
        and os.environ.get("IN_NIX_SHELL") != "impure"
    ),
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_extension_dependent_config_option():
    con = duckdb.connect()
    con.install_extension("httpfs")
    del con

    ibis.duckdb.connect(s3_endpoint="aws.example.com")
