from __future__ import annotations  # noqa: INP001

from inspect import signature
from typing import Any

import pytest
from pytest import param

from ibis.backends.tests.signature.typecheck import compatible


def a1(posarg: int): ...


def b1(posarg: str): ...


def a2(posarg: int, **kwargs: Any): ...
def b2(posarg: str, **kwargs): ...


def a3(posarg: int, other_kwarg: bool = True, **kwargs: Any): ...
def b3(posarg: str, **kwargs: Any): ...


def a4(posarg: int, other_kwarg=True, **kwargs: Any): ...
def b4(posarg: str, **kwargs: Any): ...


def a5(posarg: int, /): ...
def b5(posarg2: str, /): ...


@pytest.mark.parametrize(
    "a, b, check_annotations",
    [
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None: ...,  # noqa: ARG005
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,  # noqa: ARG005
            True,
            id="swapped kwarg order",
        ),
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,  # noqa: ARG005
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,  # noqa: ARG005
            True,
            id="swapped kwarg order w/extra kwarg first",
        ),
        param(
            lambda posarg, *, kwarg2=None, kwarg1=None: ...,  # noqa: ARG005
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,  # noqa: ARG005
            True,
            id="swapped kwarg order w/extra kwarg second",
        ),
        param(
            a1,
            b1,
            False,
            id="annotations diff types w/out anno check",
        ),
        param(
            a2,
            b3,
            False,
            id="annotations different but parity in annotations",
        ),
        param(
            a3,
            b3,
            False,
            id="annotations different but parity in annotations for matching kwargs",
        ),
        param(
            a4,
            b4,
            False,
            id="annotations different but parity in annotations for matching kwargs",
        ),
        param(
            a2,
            b2,
            False,
            id="annotations different, no anno check, but missing annotation",
        ),
    ],
)
def test_sigs_compatible(a, b, check_annotations):
    sig_a, sig_b = signature(a), signature(b)
    assert compatible(sig_a, sig_b, check_annotations=check_annotations)


@pytest.mark.parametrize(
    "a, b, check_annotations",
    [
        param(
            lambda posarg, /, *, kwarg2=None, kwarg1=None: ...,  # noqa: ARG005
            lambda posarg, *, kwarg1=None, kwarg2=None, kwarg3=None: ...,  # noqa: ARG005
            True,
            id="one positional only",
        ),
        param(
            lambda posarg, *, kwarg1=None, kwarg2=None: ...,  # noqa: ARG005
            lambda posarg, kwarg1=None, kwarg2=None: ...,  # noqa: ARG005
            True,
            id="not kwarg only",
        ),
        param(
            a1,
            b1,
            True,
            id="annotations diff types w/anno check",
        ),
        param(
            a2,
            b3,
            True,
            id="annotations different but parity in annotations",
        ),
        param(
            a5,
            b5,
            False,
            id="names different, but positional only",
        ),
    ],
)
def test_sigs_incompatible(a, b, check_annotations):
    sig_a, sig_b = signature(a), signature(b)
    assert not compatible(sig_a, sig_b, check_annotations=check_annotations)
