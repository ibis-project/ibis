from __future__ import annotations

from typing import Any

from public import public


# TODO(kszucs): it was a subclass of Singleton
@public
class DataShape:
    ndim: int
    SCALAR: Scalar = None
    COLUMNAR: Columnar = None

    def is_scalar(self) -> bool:
        return self.ndim == 0

    def is_columnar(self) -> bool:
        return self.ndim == 1

    def is_tabular(self) -> bool:
        return self.ndim == 2

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim < other.ndim

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim <= other.ndim

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim == other.ndim

    def __hash__(self) -> int:
        return hash((self.__class__, self.ndim))


@public
class Scalar(DataShape):
    ndim = 0


@public
class Columnar(DataShape):
    ndim = 1


@public
class Tabular(DataShape):
    ndim = 2


# for backward compat
scalar = DataShape.SCALAR = Scalar()
columnar = DataShape.COLUMNAR = Columnar()
tabular = DataShape.TABULAR = Tabular()


public(Any=DataShape)
