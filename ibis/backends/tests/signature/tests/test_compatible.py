from __future__ import annotations  # noqa: INP001

from inspect import signature

import pytest
from pytest import param

from ibis.backends.tests.signature.typecheck import compatible


@pytest.mark.parametrize(
    "a, b, expected",
    [
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None: ...,
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,
            True,
            id="swapped kwarg order",
        ),
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,
            True,
            id="swapped kwarg order w/extra kwarg first",
        ),
        param(
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,
            True,
            id="swapped kwarg order w/extra kwarg second",
        ),
        param(
            lambda posarg, /, *, kwarg2=None, kwarg1=None: ...,
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,
            False,
            id="one positional only",
        ),
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None: ...,
            lambda posarg, kwarg1=None, kwarg2=None: ...,
            False,
            id="not kwarg only",
        ),
    ],
)
def test_sigs_compatible(a, b, expected):
    sig_a, sig_b = signature(a), signature(b)
    assert compatible(sig_a, sig_b) == expected
