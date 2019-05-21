import operator

import six

import toolz

import ibis
import ibis.common as com

from ibis.pandas.core import execute


def compute_sort_key(key, data, scope=None, **kwargs):
    by = key.to_expr()
    try:
        if isinstance(by, six.string_types):
            return by, None
        return by.get_name(), None
    except com.ExpressionError:
        new_scope = {t: data for t in by.op().root_tables()}
        new_column = execute(by, scope=toolz.merge(scope, new_scope), **kwargs)
        name = ibis.util.guid()
        new_column.name = name
        return name, new_column


def compute_sorted_frame(df, order_by, group_by=(), **kwargs):
    computed_sort_keys = []
    sort_keys = list(toolz.concatv(group_by, order_by))
    ascending = [getattr(key.op(), 'ascending', True) for key in sort_keys]
    new_columns = {}

    for i, key in enumerate(map(operator.methodcaller('op'), sort_keys)):
        computed_sort_key, temporary_column = compute_sort_key(
            key, df, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)

        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(
        computed_sort_keys, ascending=ascending, kind='mergesort'
    )
    # TODO: we'll eventually need to return this frame with the temporary
    # columns and drop them in the caller (maybe using post_execute?)
    ngrouping_keys = len(group_by)
    return (
        result,
        computed_sort_keys[:ngrouping_keys],
        computed_sort_keys[ngrouping_keys:],
    )
