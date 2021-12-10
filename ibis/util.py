"""Ibis util functions."""
import collections
import functools
import itertools
import logging
import operator
import os
import types
import warnings
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
    """Return a uuid4 hexadecimal value.

    Returns
    -------
    string
    """
    return uuid4().hex


def indent(text: str, spaces: int) -> str:
    """Apply an indentation using the given spaces into the given text.

    Parameters
    ----------
    text : string
    spaces : string

    Returns
    -------
    string
    """
    prefix = ' ' * spaces
    return ''.join(prefix + line for line in text.splitlines(True))


def is_one_of(values: Sequence[T], t: Type[U]) -> Iterator[bool]:
    """Check if the type of each value is the same of the given type.

    Parameters
    ----------
    values : list or tuple
    t : type

    Returns
    -------
    tuple
    """
    return (isinstance(x, t) for x in values)


any_of = toolz.compose(any, is_one_of)
all_of = toolz.compose(all, is_one_of)


def promote_list(val: Union[V, List[V]]) -> List[V]:
    """Ensure that the value is a list.

    Parameters
    ----------
    val : list or object

    Returns
    -------
    list
    """
    if not isinstance(val, list):
        val = [val]
    return val


def is_function(v: Any) -> bool:
    """Check if the given object is a function.

    Parameters
    ----------
    v : object

    Returns
    -------
    bool
    """
    return isinstance(v, (types.FunctionType, types.LambdaType))


def adjoin(space: int, *lists: Sequence[str]) -> str:
    """Glue together two sets of strings using `space`.

    Parameters
    ----------
    space : int
    lists : list or tuple

    Returns
    -------
    string
    """
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
    """Log `msg` using ``options.verbose_log`` if set, otherwise ``print``.

    Parameters
    ----------
    msg : string
    """
    if options.verbose:
        (options.verbose_log or print)(msg)


def approx_equal(a: Real, b: Real, eps: Real):
    """Return whether the difference between `a` and `b` is less than `eps`.

    Parameters
    ----------
    a : real
    b : real
    eps : real

    Raises
    ------
    AssertionError
    """
    assert abs(a - b) < eps


def safe_index(elements: Sequence[int], value: int) -> int:
    """Find the location of `value` in `elements`.

    Return -1 if `value` is not found instead of raising ``ValueError``.

    Parameters
    ----------
    elements : list or tuple
    value : int
        Index of the given sequence/elements

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
    o : object
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


def convert_unit(value, unit, to, floor=True):
    """Convert a value between different units.

    Convert `value`, is assumed to be in units of `unit`, to units of `to`.
    If `floor` is true, then use floor division on `value` if necessary.

    Parameters
    ----------
    value : Union[numbers.Real, ibis.expr.types.NumericValue]
    floor : Boolean
        Flags whether or not to use floor division on `value` if necessary.

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
    if floor:
        return value // factor
    else:
        return value / factor


def get_logger(
    name: str, level: str = None, format: str = None, propagate: bool = False
) -> logging.Logger:
    """Get a logger.

    Parameters
    ----------
    name : string
    level : string
    format : string
    propagate : bool, default False

    Returns
    -------
    logging.Logger
    """
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
    """Advance the iterator n-steps ahead. If n is None, consume entirely.

    Parameters
    ----------
    iterator : list or tuple
    n : int, optional
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


def flatten_iterable(iterable):
    """Recursively flatten the iterable `iterable`."""
    if not is_iterable(iterable):
        raise TypeError("flatten is only defined for non-str iterables")

    for item in iterable:
        if is_iterable(item):
            yield from flatten_iterable(item)
        else:
            yield item


def warn_deprecated(name, *, instead, version='', stacklevel=1):
    """Warn about deprecated usage, including stacktrace, and
    what to do instead."""
    msg = f'"{name}" is deprecated'
    if version:
        msg += f' since v{version}'

    msg += f'; use {instead}.'

    warnings.warn(msg, FutureWarning, stacklevel=stacklevel + 1)


def deprecated(*, instead, version=''):
    """Decorate deprecated function to warn of usage, with stacktrace, and
    what to do instead."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(
                func.__name__, instead=instead, version=version, stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
