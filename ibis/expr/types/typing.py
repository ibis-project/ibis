from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

__all__ = ["K", "V"]

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
