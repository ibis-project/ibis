from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from public import public

from .generic import AnyColumn, AnyScalar, AnyValue, literal
from .typing import K, V

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


@public
class MapValue(AnyValue):
    pass  # noqa: E701,E302


@public
class MapScalar(AnyScalar, MapValue):
    pass  # noqa: E701,E302


@public
class MapColumn(AnyColumn, MapValue):
    pass  # noqa: E701,E302


@public
def map(
    value: Iterable[tuple[K, V]] | Mapping[K, V],
    type: str | dt.DataType | None = None,
) -> MapValue:
    """Create a map literal from a [`dict`][dict] or other mapping.

    Parameters
    ----------
    value
        the literal map value
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`.

    Returns
    -------
    MapScalar
        An expression representing a literal map (associative array with
        key/value pairs of fixed types)

    Examples
    --------
    Create a map literal from a dict with the type inferred
    >>> import ibis
    >>> t = ibis.map(dict(a=1, b=2))

    Create a map literal from a dict with the specified type
    >>> import ibis
    >>> t = ibis.map(dict(a=1, b=2), type='map<string, double>')
    """
    return literal(dict(value), type=type)
