"""Materialize-specific operations."""

from __future__ import annotations

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.expr.operations.generic import Impure


@public
class MzNow(Impure):
    """Return the logical timestamp in Materialize.

    This is Materialize's `mz_now()` function, which returns the logical time
    at which the query was executed. This is different from PostgreSQL's `now()`
    which returns the system clock time.

    Key differences from `now()`:
    - Returns logical timestamp (for streaming/incremental computation)
    - Can be used in temporal filters in materialized views
    - Value represents query execution time in Materialize's consistency model

    Best practice: When using mz_now() in temporal filters, isolate it on one side
    of the comparison for optimal incremental computation. For example:

        # Good: mz_now() > created_at + INTERVAL '1 day'
        # Bad:  mz_now() - created_at > INTERVAL '1 day'

    References
    ----------
    - Function documentation: https://materialize.com/docs/sql/functions/now_and_mz_now/
    - Idiomatic patterns: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#temporal-filters

    Returns
    -------
    mz_timestamp
        The logical timestamp as a timestamp with timezone
    """

    dtype = dt.Timestamp(timezone="UTC")
    shape = ds.scalar
