from __future__ import annotations

from typing import ForwardRef, Generic, Optional, Union

from typing_extensions import TypeVar

from ibis.common.typing import (
    DefaultTypeVars,
    Sentinel,
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
    annots = {"a": "Union[int, str]", "b": "Optional[str]"}
    hints = evaluate_annotations(annots, module_name=__name__)
    assert hints == {"a": Union[int, str], "b": Optional[str]}


def test_evaluate_annotations_with_self() -> None:
    annots = {"a": "Union[int, Self]", "b": "Optional[Self]"}
    myhint = ForwardRef(f"{__name__}.My")
    hints = evaluate_annotations(annots, module_name=__name__, class_name="My")
    assert hints == {"a": Union[int, myhint], "b": Optional[myhint]}


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
    assert get_type_params(A[int, float, str]) == {"T": int, "S": float, "U": str}
    assert get_type_params(B[int, bool]) == {"T": int, "S": bool, "U": bytes}
    assert get_type_params(C[int]) == {"T": int, "S": str, "U": bytes}
    assert get_type_params(D) == {"T": bool, "S": str, "U": bytes}


def test_get_bound_typevars() -> None:
    expected = {
        T: ("t", int),
        S: ("s", float),
        U: ("u", str),
    }
    assert get_bound_typevars(A[int, float, str]) == expected

    expected = {
        T: ("t", int),
        S: ("s", bool),
        U: ("u", bytes),
    }
    assert get_bound_typevars(B[int, bool]) == expected


def test_default_type_vars():
    T = TypeVar("T")
    U = TypeVar("U", default=float)

    class Test(DefaultTypeVars, Generic[T, U]):
        pass

    assert Test[int, float].__parameters__ == ()
    assert Test[int, float].__args__ == (int, float)

    assert Test[int].__parameters__ == ()
    assert Test[int].__args__ == (int, float)


def test_sentinel():
    class missing(metaclass=Sentinel):
        """marker for missing value"""

    class missing1(metaclass=Sentinel):
        """marker for missing value"""

    assert type(missing) is Sentinel
    expected = "<class 'ibis.common.tests.test_typing.test_sentinel.<locals>.missing'>"
    assert repr(missing) == expected
    assert missing.__name__ == "missing"
    assert missing.__doc__ == "marker for missing value"

    assert missing is missing
    assert missing is not missing1
    assert missing != missing1
    assert missing != "missing"
