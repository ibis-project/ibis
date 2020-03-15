"""Ibis random number generation expression definitions."""

import ibis.expr.operations as ops


def random():
    """
    Return a random floating point number in the range [0.0, 1.0). Similar to
    ``random.random`` in the Python standard library
    https://docs.python.org/library/random.html

    Returns
    -------
    random : random float value expression
    """
    op = ops.RandomScalar()
    return op.to_expr()
