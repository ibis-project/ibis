import collections
import functools
import itertools
import logging
import operator
import os
import types
from numbers import Real
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

import toolz

from ibis.config import options

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
V = TypeVar("V")


def guid() -> str:
    return uuid4().hex


def indent(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return ''.join(prefix + line for line in text.splitlines(True))


def is_one_of(values: Sequence[T], t: Type[U]) -> Iterator[bool]:
    return (isinstance(x, t) for x in values)


any_of = toolz.compose(any, is_one_of)
all_of = toolz.compose(all, is_one_of)


def promote_list(val: Union[V, List[V]]) -> List[V]:
    if not isinstance(val, list):
        val = [val]
    return val


def is_function(v: Any) -> bool:
    return isinstance(v, (types.FunctionType, types.LambdaType))


def adjoin(space: int, *lists: Sequence[str]) -> str:
    """Glue together two sets of strings using `space`."""
    lengths = [max(map(len, x)) + space for x in lists[:-1]]

    # not the last one
    lengths.append(max(map(len, lists[-1])))
    max_len = max(map(len, lists))
    chains = (
        itertools.chain(
            (x.ljust(length) for x in lst),
            itertools.repeat(' ' * length, max_len - len(lst)),
        )
        for lst, length in zip(lists, lengths)
    )
    return '\n'.join(map(''.join, zip(*chains)))


def log(msg: str) -> None:
    """Log `msg` using ``options.verbose_log`` if set, otherwise ``print``."""
    if options.verbose:
        (options.verbose_log or print)(msg)


def approx_equal(a: Real, b: Real, eps: Real):
    """Return whether the difference between `a` and `b` is less than `eps`.

    Parameters
    ----------
    a
    b
    eps

    Returns
    -------
    bool

    """
    assert abs(a - b) < eps


def safe_index(elements: Sequence[T], value: T):
    """Find the location of `value` in `elements`, return -1 if `value` is
    not found instead of raising ``ValueError``.

    Parameters
    ----------
    elements
    value

    Returns
    -------
    int

    Examples
    --------
    >>> sequence = [1, 2, 3]
    >>> safe_index(sequence, 2)
    1
    >>> safe_index(sequence, 4)
    -1

    """
    try:
        return elements.index(value)
    except ValueError:
        return -1


def is_iterable(o: Any) -> bool:
    """Return whether `o` is iterable and not a :class:`str` or :class:`bytes`.

    Parameters
    ----------
    o
        Any python object

    Returns
    -------
    bool

    Examples
    --------
    >>> is_iterable('1')
    False
    >>> is_iterable(b'1')
    False
    >>> is_iterable(iter('1'))
    True
    >>> is_iterable(i for i in range(1))
    True
    >>> is_iterable(1)
    False
    >>> is_iterable([])
    True

    """
    return not isinstance(o, (str, bytes)) and isinstance(
        o, collections.abc.Iterable
    )


def convert_unit(value, unit, to):
    """Convert `value`, is assumed to be in units of `unit`, to units of `to`.

    Parameters
    ----------
    value : Union[numbers.Real, ibis.expr.types.NumericValue]

    Returns
    -------
    Union[numbers.Integral, ibis.expr.types.NumericValue]

    Examples
    --------
    >>> one_second = 1000
    >>> x = convert_unit(one_second, 'ms', 's')
    >>> x
    1
    >>> one_second = 1
    >>> x = convert_unit(one_second, 's', 'ms')
    >>> x
    1000
    >>> x = convert_unit(one_second, 's', 's')
    >>> x
    1
    >>> x = convert_unit(one_second, 's', 'M')
    Traceback (most recent call last):
        ...
    ValueError: Cannot convert to or from variable length interval

    """
    # Don't do anything if from and to units are equivalent
    if unit == to:
        return value

    units = ('W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns')
    factors = (7, 24, 60, 60, 1000, 1000, 1000)

    monthly_units = ('Y', 'Q', 'M')
    monthly_factors = (4, 3)

    try:
        i, j = units.index(unit), units.index(to)
    except ValueError:
        try:
            i, j = monthly_units.index(unit), monthly_units.index(to)
            factors = monthly_factors
        except ValueError:
            raise ValueError(
                'Cannot convert to or from variable length interval'
            )

    factor = functools.reduce(operator.mul, factors[min(i, j) : max(i, j)], 1)
    assert factor > 1

    if i < j:
        return value * factor

    assert i > j
    return value // factor


def get_logger(name, level=None, format=None, propagate=False):
    logging.basicConfig()
    handler = logging.StreamHandler()

    if format is None:
        format = (
            '%(relativeCreated)6d '
            '%(name)-20s '
            '%(levelname)-8s '
            '%(threadName)-25s '
            '%(message)s'
        )
    handler.setFormatter(logging.Formatter(fmt=format))
    logger = logging.getLogger(name)
    logger.propagate = propagate
    logger.setLevel(
        level
        or getattr(logging, os.environ.get('LOGLEVEL', 'WARNING').upper())
    )
    logger.addHandler(handler)
    return logger


# taken from the itertools documentation
def consume(iterator: Iterator[T], n: Optional[int] = None) -> None:
    """Advance the iterator n-steps ahead. If n is None, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)
