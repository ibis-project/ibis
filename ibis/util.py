"""Ibis utility functions."""
from __future__ import annotations

import abc
import collections
import functools
import importlib.metadata as _importlib_metadata
import itertools
import logging
import operator
import os
import sys
import textwrap
import types
import warnings
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
)
from uuid import uuid4

import toolz

from ibis.config import options

if TYPE_CHECKING:
    import pandas as pd

    from ibis.expr import operations as ops
    from ibis.expr import types as ir

    Graph = Mapping[ops.Node, Sequence[ops.Node]]

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
V = TypeVar("V")

# https://www.compart.com/en/unicode/U+22EE
VERTICAL_ELLIPSIS = "\u22EE"
# https://www.compart.com/en/unicode/U+2026
HORIZONTAL_ELLIPSIS = "\u2026"


class frozendict(Mapping, Hashable):

    __slots__ = ("_dict", "_hash")

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = hash(tuple(self._dict.items()))

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._dict!r})"

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __hash__(self):
        return self._hash


class UnnamedMarker:
    pass


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
    text
        Text to indent
    spaces
        Number of leading spaces per line

    Returns
    -------
    str
        Indented text
    """
    prefix = ' ' * spaces
    return textwrap.indent(text, prefix=prefix)


def is_one_of(values: Sequence[T], t: type[U]) -> Iterator[bool]:
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


def promote_list(val: V | Sequence[V]) -> list[V]:
    """Ensure that the value is a list.

    Parameters
    ----------
    val : list or object

    Returns
    -------
    list
    """
    if isinstance(val, list):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val]


def is_function(v: Any) -> bool:
    """Check if the given object is a function.

    Parameters
    ----------
    v : object

    Returns
    -------
    bool
        Whether `v` is a function
    """
    return isinstance(v, (types.FunctionType, types.LambdaType))


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
def consume(iterator: Iterator[T], n: int | None = None) -> None:
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


def deprecated_msg(name, *, instead, version=''):
    msg = f'`{name}` is deprecated'
    if version:
        msg += f' as of v{version}'

    msg += f'; {instead}'
    return msg


def warn_deprecated(name, *, instead, version='', stacklevel=1):
    """Warn about deprecated usage.

    The message includes a stacktrace and what to do instead.
    """

    msg = deprecated_msg(name, instead=instead, version=version)
    warnings.warn(msg, FutureWarning, stacklevel=stacklevel + 1)


def deprecated(*, instead, version=''):
    """Decorate deprecated function to warn of usage, with stacktrace, and
    what to do instead."""

    def decorator(func):
        msg = deprecated_msg(func.__name__, instead=instead, version=version)

        docstr = func.__doc__ or ""
        first, *rest = docstr.split("\n\n", maxsplit=1)
        # count leading spaces and add them to the deprecation warning so the
        # docstring parses correctly
        leading_spaces = " " * sum(
            1
            for _ in itertools.takewhile(str.isspace, rest[0] if rest else [])
        )
        warning_doc = f'{leading_spaces}!!! warning "DEPRECATED: {msg}"'
        func.__doc__ = "\n\n".join([first, warning_doc, *rest])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(
                func.__name__, instead=instead, version=version, stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def to_op_dag(expr: ir.Expr) -> Graph:
    """Convert `expr` into a directed acyclic graph.

    Parameters
    ----------
    expr
        An ibis expression

    Returns
    -------
    Graph
        A directed acyclic graph of ibis operations
    """
    stack = [expr.op()]
    dag = {}

    while stack:
        if (node := stack.pop()) not in dag:
            dag[node] = children = node._flat_ops
            stack.extend(children)
    return dag


def get_dependents(dependencies: Graph) -> Graph:
    """Convert dependencies to dependents.

    Parameters
    ----------
    dependencies
        A mapping of [`ops.Node`][ibis.expr.operations.Node]s to a set of that
        node's `ops.Node` dependencies.

    Returns
    -------
    Graph
        A mapping of [`ops.Node`][ibis.expr.operations.Node]s to a set of that
        node's `ops.Node` dependents.
    """
    dependents = {src: [] for src in dependencies.keys()}
    for src, dests in dependencies.items():
        for dest in dests:
            dependents[dest].append(src)
    return dependents


def toposort(graph: Graph) -> Iterator[ops.Node]:
    """Topologically sort `graph` using Kahn's algorithm.

    Parameters
    ----------
    graph
        A DAG built from an ibis expression.

    Yields
    ------
    Node
        An operation node
    """
    if not graph:
        return

    dependents = get_dependents(graph)
    in_degree = {
        node: len(dependencies) for node, dependencies in graph.items()
    }
    queue = collections.deque(
        node for node, count in in_degree.items() if not count
    )
    while queue:
        dependency = queue.popleft()
        yield dependency
        for dependent in dependents[dependency]:
            in_degree[dependent] -= 1

            if not in_degree[dependent]:
                queue.append(dependent)

    if any(in_degree.values()):
        raise ValueError("cycle in expression graph")


class ToFrame(abc.ABC):
    """Interface for in-memory objects that can be converted to a DataFrame."""

    __slots__ = ()

    @abc.abstractmethod
    def to_frame(self) -> pd.DataFrame:
        ...


def backend_entry_points() -> list[_importlib_metadata.EntryPoint]:
    """Get the list of installed `ibis.backend` entrypoints"""

    if sys.version_info < (3, 10):
        eps = _importlib_metadata.entry_points()["ibis.backends"]
    else:
        eps = _importlib_metadata.entry_points(group="ibis.backends")
    return sorted(eps)
