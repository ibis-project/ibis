from __future__ import annotations

import ibis.expr.types as ir  # noqa: TCH001
from ibis.common.grounds import Concrete


class Watermark(Concrete):
    time_col: str
    allowed_delay: ir.IntervalScalar
