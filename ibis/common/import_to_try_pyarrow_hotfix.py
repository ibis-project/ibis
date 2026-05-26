from __future__ import annotations

import os
import warnings


def try_apply_pyarrow_hotfix() -> None:
    """Apply `pyarrow-hotfix` for vulnerable pyarrow versions when available.

    This is a patch for https://nvd.nist.gov/vuln/detail/cve-2023-47248.
    See https://github.com/ibis-project/ibis/pull/11977 for discussion.
    """
    if os.environ.get("IBIS_PYARROW_HOTFIX_SKIP"):
        return

    try:
        import pyarrow as pa
    except ImportError:
        # Not even installed, so no need to patch.
        return

    if tuple(int(x) for x in pa.__version__.split(".")[:3]) > (14, 0, 0):
        # New enough version, so no need to patch.
        return

    try:
        import pyarrow_hotfix  # noqa: F401
    except ImportError:
        warnings.warn(
            "You are using an old version of pyarrow "
            f"({pa.__version__}) that is vulnerable to "
            "https://nvd.nist.gov/vuln/detail/cve-2023-47248. "
            "Either upgrade pyarrow to version >14, or install "
            "`pyarrow-hotfix` to patch the old version. "
            "Set `IBIS_PYARROW_HOTFIX_SKIP=true` to skip this autopatch "
            "and silence this warning.",
            stacklevel=2,
        )


try_apply_pyarrow_hotfix()
