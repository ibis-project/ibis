from __future__ import annotations

from typing import TYPE_CHECKING

from ibis.common.collections import frozendict  # noqa: TCH001
from ibis.common.graph import Node
from ibis.common.grounds import Concrete

if TYPE_CHECKING:
    from typing_extensions import Self


class MyNode(Node, Concrete):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]
    e: Self
    f: tuple[Self, ...]
