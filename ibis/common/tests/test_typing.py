from __future__ import annotations

from typing import Generic, Optional, TypeVar, Union

from ibis.common.typing import (
    evaluate_annotations,
    get_bound_typevars,
    get_type_hints,
    get_type_params,
)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


class My(Generic[T, S, U]):
    a: T
    b: S
    c: str

    @property
    def d(self) -> Optional[str]:
        ...

    @property
    def e(self) -> U:  # type: ignore
        ...


class MyChild(My):
    ...


def example(a: int, b: str) -> str:  # type: ignore
    ...


def test_evaluate_annotations() -> None:
    annotations = {"a": "Union[int, str]", "b": "Optional[str]"}
    hints = evaluate_annotations(annotations, module_name=__name__)
    assert hints == {"a": Union[int, str], "b": Optional[str]}


def test_get_type_hints() -> None:
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(My, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    hints = get_type_hints(MyChild, include_properties=True)
    assert hints == {"a": T, "b": S, "c": str, "d": Optional[str], "e": U}

    # test that we don't actually mutate the My.__annotations__
    hints = get_type_hints(My)
    assert hints == {"a": T, "b": S, "c": str}

    hints = get_type_hints(example)
    assert hints == {"a": int, "b": str, "return": str}

    hints = get_type_hints(example, include_properties=True)
    assert hints == {"a": int, "b": str, "return": str}


class A(Generic[T, S, U]):
    a: int
    b: str

    t: T
    s: S

    @property
    def u(self) -> U:  # type: ignore
        ...


class B(A[T, S, bytes]):
    ...


class C(B[T, str]):
    ...


class D(C[bool]):
    ...


def test_get_type_params() -> None:
    assert get_type_params(A[int, float, str]) == {'T': int, 'S': float, 'U': str}
    assert get_type_params(B[int, bool]) == {'T': int, 'S': bool, 'U': bytes}
    assert get_type_params(C[int]) == {'T': int, 'S': str, 'U': bytes}
    assert get_type_params(D) == {'T': bool, 'S': str, 'U': bytes}


def test_get_bound_typevars() -> None:
    assert get_bound_typevars(A[int, float, str]) == {'t': int, 's': float, 'u': str}
    assert get_bound_typevars(B[int, bool]) == {'t': int, 's': bool, 'u': bytes}
