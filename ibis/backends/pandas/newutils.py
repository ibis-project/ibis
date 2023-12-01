from __future__ import annotations

import itertools
import operator
from collections.abc import Sized
from typing import Callable

import numpy as np
import pandas as pd


def asframe(values: dict | tuple):
    if isinstance(values, dict):
        names, values = zip(*values.items())
    elif isinstance(values, tuple):
        names = [f"_{i}" for i in range(len(values))]
    else:
        raise TypeError(f"values must be a dict, list, or tuple; got {type(values)}")

    size = 1
    all_scalars = True
    for v in values:
        if isinstance(v, pd.Series):
            size = len(v)
            all_scalars = False
            break

    columns = []
    for v in values:
        if isinstance(v, pd.Series):
            pass
        elif isinstance(v, (list, np.ndarray)):
            v = pd.Series(itertools.repeat(np.array(v), size))
        else:
            v = pd.Series(np.repeat(v, size))
        columns.append(v)

    return pd.concat(columns, axis=1, keys=names), all_scalars


def rowwise(_func: Callable, _values: dict | tuple, **kwargs):
    # dealing with a collection of series objects
    df, all_scalars = asframe(_values)
    result = df.apply(_func, axis=1, **kwargs)
    # if astype is not None:
    #     result = result.astype(astype)
    return result.iloc[0] if all_scalars else result


def columnwise(_func: Callable, _values: dict | tuple, **kwargs):
    df, all_scalars = asframe(_values)
    result = _func(df, **kwargs)
    return result.iloc[0] if all_scalars else result


# TODO(kszucs): change kwarg to _values everywhere, should follow the API of r
def serieswise(_func, **kwargs):
    (key, value), *rest = kwargs.items()
    if isinstance(value, pd.Series):
        # dealing with a single series object
        return _func(**kwargs)
    else:
        # dealing with a single scalar object
        value = pd.Series([value])
        kwargs = {key: value, **dict(rest)}
        return _func(**kwargs).iloc[0]


def elementwise(_func, **kwargs):
    value = kwargs.pop(next(iter(kwargs)))
    if isinstance(value, pd.Series):
        # dealing with a single series object
        return value.map(_func, **kwargs)
    else:
        # dealing with a single scalar object
        return _func(value)  # , **kwargs)


####################### STRING FUNCTIONS #######################


def _sql_like_to_regex(pattern, escape):
    cur_i = 0
    pattern_length = len(pattern)

    while cur_i < pattern_length:
        nxt_i = cur_i + 1

        cur = pattern[cur_i]
        nxt = pattern[nxt_i] if nxt_i < pattern_length else None

        skip = 1

        if nxt is not None and escape is not None and cur == escape:
            yield nxt
            skip = 2
        elif cur == "%":
            yield ".*"
        elif cur == "_":
            yield "."
        else:
            yield cur

        cur_i += skip


def sql_like_to_regex(pattern: str, escape: str | None = None) -> str:
    """Convert a SQL `LIKE` pattern to an equivalent Python regular expression.

    Parameters
    ----------
    pattern
        A LIKE pattern with the following semantics:
        * `%` matches zero or more characters
        * `_` matches exactly one character
        * To escape `%` and `_` (or to match the `escape` parameter
          itself), prefix the desired character with `escape`.
    escape
        Escape character

    Returns
    -------
    str
        A regular expression pattern equivalent to the input SQL `LIKE` pattern.

    Examples
    --------
    >>> sql_like_to_regex("6%")  # default is to not escape anything
    '^6.*$'
    >>> sql_like_to_regex("6^%", escape="^")
    '^6%$'
    >>> sql_like_to_regex("6_")
    '^6.$'
    >>> sql_like_to_regex("6/_", escape="/")
    '^6_$'
    >>> sql_like_to_regex("%abc")  # any string ending with "abc"
    '^.*abc$'
    >>> sql_like_to_regex("abc%")  # any string starting with "abc"
    '^abc.*$'
    """
    return f"^{''.join(_sql_like_to_regex(pattern, escape))}$"
