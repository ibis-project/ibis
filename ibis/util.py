from __future__ import print_function

import collections
import functools
import logging
import operator
import os
import types

import six

import toolz

import ibis.compat as compat
from ibis.config import options


def guid():
    try:
        from ibis.comms import uuid4_hex
        return uuid4_hex()
    except ImportError:
        from uuid import uuid4
        guid = uuid4()
        return guid.hex if not compat.PY2 else guid.get_hex()


def indent(text, spaces):
    prefix = ' ' * spaces
    return ''.join(prefix + line for line in text.splitlines(True))


def is_one_of(values, t):
    return (isinstance(x, t) for x in values)


any_of = toolz.compose(any, is_one_of)
all_of = toolz.compose(all, is_one_of)


def promote_list(val):
    if not isinstance(val, list):
        val = [val]
    return val


class IbisSet(object):

    def __init__(self, keys=None):
        self.keys = keys or []

    @classmethod
    def from_list(cls, keys):
        return IbisSet(keys)

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def add(self, obj):
        self.keys.append(obj)


class IbisMap(object):

    def __init__(self):
        self.keys = []
        self.values = []

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def set(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def get(self, key):
        for k, v in zip(self.keys, self.values):
            if key.equals(k):
                return v
        raise KeyError(key)


def is_function(v):
    return isinstance(v, (types.FunctionType, types.LambdaType))


def adjoin(space, *lists):
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    Brought over from from pandas
    """
    out_lines = []
    newLists = []
    lengths = [max(map(len, x)) + space for x in lists[:-1]]

    # not the last one
    lengths.append(max(map(len, lists[-1])))

    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = [x.ljust(lengths[i]) for x in lst]
        nl.extend([' ' * lengths[i]] * (maxLen - len(lst)))
        newLists.append(nl)
    toJoin = zip(*newLists)
    for lines in toJoin:
        out_lines.append(_join_unicode(lines))
    return _join_unicode(out_lines, sep='\n')


def _join_unicode(lines, sep=''):
    try:
        return sep.join(lines)
    except UnicodeDecodeError:
        sep = compat.unicode_type(sep)
        return sep.join([x.decode('utf-8') if isinstance(x, str) else x
                         for x in lines])


def log(msg):
    if options.verbose:
        (options.verbose_log or print)(msg)


def approx_equal(a, b, eps):
    """Return whether the difference between `a` and `b` is less than `eps`.

    Parameters
    ----------
    a : numbers.Real
    b : numbers.Real
    eps : numbers.Real

    Returns
    -------
    are_diff : bool
    """
    assert abs(a - b) < eps


def implements(f):
    # TODO: is this any different from functools.wraps?
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator


def safe_index(elements, value):
    """Find the location of `value` in `elements`, return -1 if `value` is
    not found instead of raising ``ValueError``.

    Parameters
    ----------
    elements : Sequence
    value : object

    Returns
    -------
    location : object

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


def is_iterable(o):
    """Return whether `o` is a non-string iterable.

    Parameters
    ----------
    o : object
        Any python object

    Returns
    -------
    is_seq : bool

    Examples
    --------
    >>> x = '1'
    >>> is_iterable(x)
    False
    >>> is_iterable(iter(x))
    True
    >>> is_iterable(i for i in range(1))
    True
    >>> is_iterable(1)
    False
    >>> is_iterable([])
    True
    """
    return (not isinstance(o, six.string_types) and
            isinstance(o, collections.Iterable))


def convert_unit(value, unit, to):
    """Convert `value`--which is assumed to be in units of `unit`--to units of
    `to`.

    Parameters
    ----------
    value : Union[numbers.Real, ibis.expr.types.NumericValue]

    Returns
    -------
    result : Union[numbers.Integral, ibis.expr.types.NumericValue]

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

    factor = functools.reduce(operator.mul, factors[min(i, j):max(i, j)], 1)
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
        level or getattr(
            logging, os.environ.get('LOGLEVEL', 'WARNING').upper()))
    logger.addHandler(handler)
    return logger
