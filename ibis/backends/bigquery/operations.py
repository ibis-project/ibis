"""Ibis operations specific to BigQuery."""

from __future__ import annotations

import ibis.expr.operations as ops


class BigQueryUDFNode(ops.ValueOp):
    """Represents use of a UDF."""
