import itertools
import operator
from functools import reduce

import numpy as np
import pandas as pd
import regex as re
import toolz
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops
import ibis.util

from ..core import integer_types, scalar_types
from ..dispatch import execute_node


@execute_node.register(ops.StringLength, pd.Series)
def execute_string_length_series(op, data, **kwargs):
    return data.str.len().astype('int32')


@execute_node.register(ops.Substring, pd.Series, integer_types, integer_types)
def execute_substring_int_int(op, data, start, length, **kwargs):
    return data.str[start : start + length]


@execute_node.register(ops.Substring, pd.Series, pd.Series, integer_types)
def execute_substring_series_int(op, data, start, length, **kwargs):
    return execute_substring_series_series(
        op, data, start, pd.Series(np.repeat(length, len(start))), **kwargs
    )


@execute_node.register(ops.Substring, pd.Series, integer_types, pd.Series)
def execute_string_substring_int_series(op, data, start, length, **kwargs):
    return execute_substring_series_series(
        op, data, pd.Series(np.repeat(start, len(length))), length, **kwargs
    )


@execute_node.register(ops.Substring, pd.Series, pd.Series, pd.Series)
def execute_substring_series_series(op, data, start, length, **kwargs):
    end = start + length

    def iterate(value, start_iter=start.values.flat, end_iter=end.values.flat):
        begin = next(start_iter)
        end = next(end_iter)
        if (begin is not None and pd.isnull(begin)) or (
            end is not None and pd.isnull(end)
        ):
            return None
        return value[begin:end]

    return data.map(iterate)


@execute_node.register(ops.Strip, pd.Series)
def execute_string_strip(op, data, **kwargs):
    return data.str.strip()


@execute_node.register(ops.LStrip, pd.Series)
def execute_string_lstrip(op, data, **kwargs):
    return data.str.lstrip()


@execute_node.register(ops.RStrip, pd.Series)
def execute_string_rstrip(op, data, **kwargs):
    return data.str.rstrip()


@execute_node.register(
    ops.LPad, pd.Series, (pd.Series,) + integer_types, (pd.Series, str)
)
def execute_string_lpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='left', fillchar=pad)


@execute_node.register(
    ops.RPad, pd.Series, (pd.Series,) + integer_types, (pd.Series, str)
)
def execute_string_rpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='right', fillchar=pad)


@execute_node.register(ops.Reverse, pd.Series)
def execute_string_reverse(op, data, **kwargs):
    return data.str[::-1]


@execute_node.register(ops.Lowercase, pd.Series)
def execute_string_lower(op, data, **kwargs):
    return data.str.lower()


@execute_node.register(ops.Uppercase, pd.Series)
def execute_string_upper(op, data, **kwargs):
    return data.str.upper()


@execute_node.register(ops.Capitalize, pd.Series)
def execute_string_capitalize(op, data, **kwargs):
    return data.str.capitalize()


@execute_node.register(ops.Repeat, pd.Series, (pd.Series,) + integer_types)
def execute_string_repeat(op, data, times, **kwargs):
    return data.str.repeat(times)


@execute_node.register(
    ops.StringFind,
    pd.Series,
    (pd.Series, str),
    (pd.Series, type(None)) + integer_types,
    (pd.Series, type(None)) + integer_types,
)
def execute_string_contains(op, data, needle, start, end, **kwargs):
    return data.str.find(needle, start, end)


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
        elif cur == '%':
            yield '.*'
        elif cur == '_':
            yield '.'
        else:
            yield cur

        cur_i += skip


def sql_like_to_regex(pattern, escape=None):
    """Convert a SQL LIKE pattern to an equivalent Python regular expression.

    Parameters
    ----------
    pattern : str
        A LIKE pattern with the following semantics:
        * ``%`` matches zero or more characters
        * ``_`` matches exactly one character
        * To escape ``%`` and ``_`` (or to match the `escape` parameter
          itself), prefix the desired character with `escape`.

    Returns
    -------
    new_pattern : str
        A regular expression pattern equivalent to the input SQL LIKE pattern.

    Examples
    --------
    >>> sql_like_to_regex('6%')  # default is to not escape anything
    '^6.*$'
    >>> sql_like_to_regex('6^%', escape='^')
    '^6%$'
    >>> sql_like_to_regex('6_')
    '^6.$'
    >>> sql_like_to_regex('6/_', escape='/')
    '^6_$'
    >>> sql_like_to_regex('%abc')  # any string ending with "abc"
    '^.*abc$'
    >>> sql_like_to_regex('abc%')  # any string starting with "abc"
    '^abc.*$'
    """
    return '^{}$'.format(''.join(_sql_like_to_regex(pattern, escape)))


@execute_node.register(ops.StringSQLLike, pd.Series, str, (str, type(None)))
def execute_string_like_series_string(op, data, pattern, escape, **kwargs):
    new_pattern = re.compile(sql_like_to_regex(pattern, escape=escape))
    return data.map(
        lambda x, pattern=new_pattern: pattern.search(x) is not None
    )


@execute_node.register(ops.StringSQLLike, SeriesGroupBy, str, str)
def execute_string_like_series_groupby_string(
    op, data, pattern, escape, **kwargs
):
    return execute_string_like_series_string(
        op, data.obj, pattern, escape, **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(
    ops.GroupConcat, pd.Series, str, (pd.Series, type(None))
)
def execute_group_concat_series_mask(
    op, data, sep, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        lambda series, sep=sep: sep.join(series.values),
    )


@execute_node.register(ops.GroupConcat, SeriesGroupBy, str, type(None))
def execute_group_concat_series_gb(
    op, data, sep, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(
        data, lambda data, sep=sep: sep.join(data.values.astype(str))
    )


@execute_node.register(ops.GroupConcat, SeriesGroupBy, str, SeriesGroupBy)
def execute_group_concat_series_gb_mask(
    op, data, sep, mask, aggcontext=None, **kwargs
):
    def method(series, sep=sep):
        return sep.join(series.values.astype(str))

    return aggcontext.agg(
        data,
        lambda data, mask=mask.obj, method=method: method(
            data[mask[data.index]]
        ),
    )


@execute_node.register(ops.StringAscii, pd.Series)
def execute_string_ascii(op, data, **kwargs):
    return data.map(ord).astype('int32')


@execute_node.register(ops.StringAscii, SeriesGroupBy)
def execute_string_ascii_group_by(op, data, **kwargs):
    return execute_string_ascii(op, data, **kwargs).groupby(
        data.grouper.groupings
    )


@execute_node.register(ops.RegexSearch, pd.Series, str)
def execute_series_regex_search(op, data, pattern, **kwargs):
    return data.map(
        lambda x, pattern=re.compile(pattern): pattern.search(x) is not None
    )


@execute_node.register(ops.RegexSearch, SeriesGroupBy, str)
def execute_series_regex_search_gb(op, data, pattern, **kwargs):
    return execute_series_regex_search(
        op, data, getattr(pattern, 'obj', pattern), **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(
    ops.RegexExtract, pd.Series, (pd.Series, str), integer_types
)
def execute_series_regex_extract(op, data, pattern, index, **kwargs):
    def extract(x, pattern=re.compile(pattern), index=index):
        match = pattern.match(x)
        if match is not None:
            return match.group(index) or np.nan
        return np.nan

    extracted = data.apply(extract)
    return extracted


@execute_node.register(ops.RegexExtract, SeriesGroupBy, str, integer_types)
def execute_series_regex_extract_gb(op, data, pattern, index, **kwargs):
    return execute_series_regex_extract(
        op, data.obj, pattern, index, **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(ops.RegexReplace, pd.Series, str, str)
def execute_series_regex_replace(op, data, pattern, replacement, **kwargs):
    def replacer(x, pattern=re.compile(pattern)):
        return pattern.sub(replacement, x)

    return data.apply(replacer)


@execute_node.register(ops.RegexReplace, SeriesGroupBy, str, str)
def execute_series_regex_replace_gb(op, data, pattern, replacement, **kwargs):
    return execute_series_regex_replace(
        data.obj, pattern, replacement, **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(ops.Translate, pd.Series, pd.Series, pd.Series)
def execute_series_translate_series_series(
    op, data, from_string, to_string, **kwargs
):
    to_string_iter = iter(to_string)
    table = from_string.apply(
        lambda x, y: str.maketrans(x, y=next(y)), args=(to_string_iter,)
    )
    return data.str.translate(table)


@execute_node.register(ops.Translate, pd.Series, pd.Series, str)
def execute_series_translate_series_scalar(
    op, data, from_string, to_string, **kwargs
):
    table = from_string.map(lambda x, y=to_string: str.maketrans(x=x, y=y))
    return data.str.translate(table)


@execute_node.register(ops.Translate, pd.Series, str, pd.Series)
def execute_series_translate_scalar_series(
    op, data, from_string, to_string, **kwargs
):
    table = to_string.map(lambda y, x=from_string: str.maketrans(x=x, y=y))
    return data.str.translate(table)


@execute_node.register(ops.Translate, pd.Series, str, str)
def execute_series_translate_scalar_scalar(
    op, data, from_string, to_string, **kwargs
):
    return data.str.translate(str.maketrans(from_string, to_string))


@execute_node.register(ops.StrRight, pd.Series, integer_types)
def execute_series_right(op, data, nchars, **kwargs):
    return data.str[-nchars:]


@execute_node.register(ops.StrRight, SeriesGroupBy, integer_types)
def execute_series_right_gb(op, data, nchars, **kwargs):
    return execute_series_right(op, data.obj, nchars).groupby(
        data.grouper.groupings
    )


@execute_node.register(ops.StringJoin, (pd.Series, str), list)
def execute_series_join_scalar_sep(op, sep, data, **kwargs):
    return reduce(lambda x, y: x + sep + y, data)


def haystack_to_series_of_lists(haystack, index=None):
    if index is None:
        index = toolz.first(
            piece.index for piece in haystack if hasattr(piece, 'index')
        )
    pieces = reduce(
        operator.add,
        (
            pd.Series(getattr(piece, 'values', piece), index=index).map(
                ibis.util.promote_list
            )
            for piece in haystack
        ),
    )
    return pieces


@execute_node.register(ops.FindInSet, pd.Series, list)
def execute_series_find_in_set(op, needle, haystack, **kwargs):
    pieces = haystack_to_series_of_lists(haystack, index=needle.index)
    return pieces.map(
        lambda elements, needle=needle, index=itertools.count(): (
            ibis.util.safe_index(elements, needle.iat[next(index)])
        )
    )


@execute_node.register(ops.FindInSet, SeriesGroupBy, list)
def execute_series_group_by_find_in_set(op, needle, haystack, **kwargs):
    pieces = [getattr(piece, 'obj', piece) for piece in haystack]
    return execute_series_find_in_set(
        op, needle.obj, pieces, **kwargs
    ).groupby(needle.grouper.groupings)


@execute_node.register(ops.FindInSet, scalar_types, list)
def execute_string_group_by_find_in_set(op, needle, haystack, **kwargs):
    # `list` could contain series, series groupbys, or scalars
    # mixing series and series groupbys is not allowed
    series_in_haystack = [
        type(piece)
        for piece in haystack
        if isinstance(piece, (pd.Series, SeriesGroupBy))
    ]

    if not series_in_haystack:
        return ibis.util.safe_index(haystack, needle)

    try:
        (collection_type,) = frozenset(map(type, series_in_haystack))
    except ValueError:
        raise ValueError('Mixing Series and SeriesGroupBy is not allowed')

    pieces = haystack_to_series_of_lists(
        [getattr(piece, 'obj', piece) for piece in haystack]
    )

    result = pieces.map(toolz.flip(ibis.util.safe_index)(needle))
    if issubclass(collection_type, pd.Series):
        return result

    assert issubclass(collection_type, SeriesGroupBy)

    return result.groupby(
        toolz.first(
            piece.grouper.groupings
            for piece in haystack
            if hasattr(piece, 'grouper')
        )
    )
