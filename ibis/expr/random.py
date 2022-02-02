"""Ibis random number generation expression definitions."""

import ibis.expr.operations as ops
import ibis.expr.types as ir


def random() -> ir.FloatingScalar:
    """Return a random floating point number in the range [0.0, 1.0).

    Similar to [`random.random`][random.random] in the Python standard library.

    Returns
    -------
    FloatingScalar
        Random float value expression
    """
    op = ops.RandomScalar()
    return op.to_expr()
