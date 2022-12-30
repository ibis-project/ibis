"""Define types for annotation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import pandas as pd

    # Time context types
    TimeContext = Tuple[pd.Timestamp, pd.Timestamp]
