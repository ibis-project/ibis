"""Ibis operations specific to BigQuery."""

import ibis.expr.operations as ops


class BigQueryUDFNode(ops.ValueOp):
    """Represents use of a UDF."""
