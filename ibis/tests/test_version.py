from pkg_resources import parse_version
from pkg_resources.extern.packaging.version import Version

import ibis


def test_version():
    assert isinstance(parse_version(ibis.__version__), Version)
