import os

import pytest
from pkg_resources import parse_version
from pkg_resources.extern.packaging.version import Version

import ibis


@pytest.mark.skipif(
    bool(os.environ.get("AZURECI")),
    reason="Testing import time on CI is flaky due to machine variance",
)
def test_import_time():
    pb = pytest.importorskip("plumbum")
    lines = ["from timeit import timeit", "print(timeit('import ibis'))"]
    delta = float(pb.cmd.python["-c", "; ".join(lines)]().strip())
    assert delta < 2.0


def test_version():
    assert isinstance(parse_version(ibis.__version__), Version)
