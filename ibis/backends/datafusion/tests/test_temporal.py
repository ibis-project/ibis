from __future__ import annotations

from operator import methodcaller

import pytest
from pytest import param

import ibis


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        param(
            methodcaller("hour"),
            14,
            id="hour",
        ),
        param(
            methodcaller("minute"),
            48,
            id="minute",
        ),
        param(
            methodcaller("second"),
            5,
            id="second",
        ),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
        ),
    ],
)
def test_time_extract_literal(con, func, expected):
    value = ibis.time("14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected
