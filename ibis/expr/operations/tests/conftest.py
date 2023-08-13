from __future__ import annotations

import sys

collect_ignore: list[str] = []
if sys.version_info < (3, 10):
    collect_ignore.append("test_core_py310.py")
