from __future__ import annotations

from packaging.version import Version
from packaging.version import parse as parse_version

import ibis


def test_version():
    assert isinstance(parse_version(ibis.__version__), Version)
