import os

from pkg_resources import parse_version
from pkg_resources.extern.packaging.version import Version

import pytest

import ibis


@pytest.mark.skipif(
    bool(os.environ.get('CIRCLECI', None)),
    reason='Testing import time on CI is flaky due to VM variance',
)
def test_import_time():
    sh = pytest.importorskip('sh')

    lines = [
        'from timeit import timeit',
        "print(timeit('import ibis'))",
    ]

    delta = float(str(sh.python(c='; '.join(lines))))
    assert delta < 2.0


def test_version():
    assert isinstance(parse_version(ibis.__version__), Version)
