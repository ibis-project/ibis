from __future__ import annotations

import warnings

warnings.warn(
    "ibis.expr.selectors is deprecated and will be removed in 6.0; use import ibis.selectors or from ibis.selectors import ..."
)
del warnings

from ibis.selectors import *  # noqa: F403, E402
