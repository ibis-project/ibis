from __future__ import annotations

import pytest

from ibis.common.annotations import attribute
from ibis.common.collections import frozendict
from ibis.common.grounds import Concrete

pytestmark = pytest.mark.benchmark


class MyObject(Concrete):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]

    @attribute
    def e(self):
        return self.a * 2

    @attribute
    def f(self):
        return self.b * 2

    @attribute
    def g(self):
        return self.c * 2


def test_concrete_construction(benchmark):
    benchmark(MyObject, 1, "2", c=(3, 4), d=frozendict(e=5, f=6))
