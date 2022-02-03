from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING, Iterable, Mapping

from public import public

from .generic import AnyColumn, AnyScalar, AnyValue, literal
from .typing import V

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


@public
def struct(
    value: Iterable[tuple[str, V]] | Mapping[str, V],
    type: str | dt.DataType | None = None,
) -> StructValue:
    """Create a struct literal from a [`dict`][dict] or other mapping.

    Parameters
    ----------
    value
        The underlying data for literal struct value
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`.

    Returns
    -------
    StructScalar
        An expression representing a literal struct (compound type with fields
        of fixed types)

    Examples
    --------
    Create a struct literal from a [`dict`][dict] with the type inferred
    >>> import ibis
    >>> t = ibis.struct(dict(a=1, b='foo'))

    Create a struct literal from a [`dict`][dict] with a specified type
    >>> t = ibis.struct(dict(a=1, b='foo'), type='struct<a: float, b: string>')
    """
    return literal(collections.OrderedDict(value), type=type)


@public
class StructValue(AnyValue):
    def __dir__(self):
        return sorted(
            frozenset(itertools.chain(dir(type(self)), self.type().names))
        )


@public
class StructScalar(AnyScalar, StructValue):
    pass  # noqa: E701,E302


@public
class StructColumn(AnyColumn, StructValue):
    pass  # noqa: E701,E302


@public
class DestructValue(AnyValue):
    """Class that represents a destruct value.

    When assigning a destruct column, the field inside this destruct column
    will be destructured and assigned to multipe columnns.
    """


@public
class DestructScalar(AnyScalar, DestructValue):
    pass


@public
class DestructColumn(AnyColumn, DestructValue):
    pass
