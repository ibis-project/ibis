from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import Self


class MapSet(Mapping):
    def _check_conflict(self, other):
        common_keys = self.keys() & other.keys()
        for key in common_keys:
            left, right = self[key], other[key]
            if left != right:
                raise ValueError(
                    f"Conflicting values for key `{key}`: {left} != {right}"
                )
        return common_keys

    def __ge__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return other.keys() == common_keys

    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __le__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return self.keys() == common_keys

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __and__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        common_keys = self._check_conflict(other)
        intersection = {k: v for k, v in self.items() if k in common_keys}
        return self.__class__(intersection)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        common_keys = self._check_conflict(other)
        difference = {k: v for k, v in self.items() if k not in common_keys}
        return self.__class__(difference)

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        self._check_conflict(other)
        union = {**self, **other}
        return self.__class__(union)

    def __xor__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        left = self - other
        right = other - self
        left._check_conflict(right)
        union = {**left, **right}
        return self.__class__(union)

    def isdisjoint(self, other: Self) -> bool:
        common_keys = self._check_conflict(other)
        return not common_keys
