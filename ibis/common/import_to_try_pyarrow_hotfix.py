from __future__ import annotations

import pyarrow as pa

if tuple(int(x) for x in pa.__version__.split(".")[:3]) < (14, 0, 1):
    try:
        import pyarrow_hotfix  # noqa: F401
    except ImportError:
        raise ImportError("pyarrow_hotfix should be installed for pyarrow<14.0.1")
