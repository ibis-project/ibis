"""Test utilities."""

import ibis
import ibis.util as util


def assert_equal(left, right):
    """Assert that two ibis objects are equal."""

    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right), 'Comparing schemas: \n{!r} !=\n{!r}'.format(
            left, right
        )
    else:
        assert left.equals(right), 'Objects unequal: {}\nvs\n{}'.format(
            repr(left), repr(right)
        )
