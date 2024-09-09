from __future__ import annotations

from typing import ForwardRef, Generic, Optional, Union

from typing_extensions import TypeVar

from ibis.common.typing import (
    DefaultTypeVars,
    Sentinel,
    evaluate_annotations,
)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


class My(Generic[T, S, U]):
    a: T
    b: S
    c: str

    @property
    def d(self) -> Optional[str]: ...

    @property
    def e(self) -> U:  # type: ignore
        ...


class MyChild(My): ...


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


class A(Generic[T, S, U]):
    a: int
    b: str

    t: T
    s: S

    @property
    def u(self) -> U:  # type: ignore
        ...


class B(A[T, S, bytes]): ...


class C(B[T, str]): ...


class D(C[bool]): ...


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
