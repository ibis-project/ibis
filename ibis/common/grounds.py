from __future__ import annotations

from koerce import Annotable, MatchError
from typing_extensions import Self


class Concrete(Annotable, immutable=True, hashable=True):
    """Enable quick equality comparisons.

    The subclasses must implement the `__equals__` method that returns a boolean
    value indicating whether the two instances are equal. This method is called
    only if the two instances are of the same type and the result is cached for
    future comparisons.

    Since the class holds a global cache of comparison results, it is important
    to make sure that the instances are not kept alive longer than necessary.
    """

    __equality_cache__ = {}

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        # type comparison should be cheap
        if type(self) is not type(other):
            return False

        id1 = id(self)
        id2 = id(other)
        try:
            return self.__equality_cache__[id1][id2]
        except KeyError:
            result = hash(self) == hash(other) and self.__args__ == other.__args__
            self.__equality_cache__.setdefault(id1, {})[id2] = result
            self.__equality_cache__.setdefault(id2, {})[id1] = result
            return result

    def __del__(self):
        id1 = id(self)
        for id2 in self.__equality_cache__.pop(id1, ()):
            eqs2 = self.__equality_cache__[id2]
            del eqs2[id1]
            if not eqs2:
                del self.__equality_cache__[id2]

    def copy(self, **overrides) -> Self:
        kwargs = dict(zip(self.__argnames__, self.__args__))
        if unknown_args := overrides.keys() - kwargs.keys():
            raise AttributeError(f"Unexpected arguments: {unknown_args}")
        kwargs.update(overrides)
        return self.__class__(**kwargs)

    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self):
        return self.__argnames__


ValidationError = SignatureValidationError = (MatchError, ValueError, TypeError)
