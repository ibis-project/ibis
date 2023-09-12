from __future__ import annotations

from decimal import Context, localcontext
from decimal import Decimal as D

import pytest

from ibis.common.numeric import normalize_decimal


@pytest.mark.parametrize(
    ("value", "precision", "scale", "expected"),
    [
        (1, None, None, D("1")),
        (1.0, None, None, D("1.0")),
        (1.0, 2, None, D("1.0")),
        (1.0, 2, 1, D("1.0")),
        (1.0, 3, 2, D("1.0")),
        (1.0, 3, 1, D("1.0")),
        (1.0, 3, 0, D("1")),
        (1.0, 2, 0, D("1")),
        (1.0, 1, 0, D("1")),
        (3.14, 3, 2, D("3.14")),
        (3.14, 10, 2, D("3.14")),
        (3.14, 10, 3, D("3.14")),
        (3.14, 10, 4, D("3.14")),
        (1234.567, 10, 4, D("1234.567")),
        (1234.567, 10, 3, D("1234.567")),
    ],
)
def test_normalize_decimal(value, precision, scale, expected):
    assert normalize_decimal(value, precision, scale) == expected


@pytest.mark.parametrize(
    ("value", "precision", "scale"),
    [
        (1.0, 2, 2),
        (1.0, 1, 1),
        (D("1.1234"), 5, 3),
        (D("1.1234"), 4, 2),
        (D("23145"), 4, 2),
        (1234.567, 10, 2),
        (1234.567, 10, 1),
        (3.14, 10, 0),
        (3.14, 3, 0),
        (3.14, 3, 1),
        (3.14, 10, 1),
    ],
)
def test_normalize_failing(value, precision, scale):
    with pytest.raises(TypeError):
        normalize_decimal(value, precision, scale)


def test_normalize_decimal_dont_truncate_precision():
    # test that the decimal context is ignored, 38 is the default precision
    for prec in [10, 30, 38]:
        with localcontext(Context(prec=prec)):
            v = "1.123456789"
            assert str(normalize_decimal(v + "0000")) == "1.123456789"

            v = v + "1" * 28
            assert len(v) == 39
            assert str(normalize_decimal(v)) == v

            # if no precision is specified, we use precision 38 for dec.normalize()
            v = v + "1"
            assert len(v) == 40
            assert str(normalize_decimal(v)) == v[:-1]

            # pass the precision explicitly
            assert str(normalize_decimal(v, precision=39)) == v

            v = v + "1" * 11
            assert len(v) == 51
            assert str(normalize_decimal(v, precision=50)) == v
            assert str(normalize_decimal(v, precision=45)) == v[:-5]
