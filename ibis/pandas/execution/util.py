import operator

import ibis
import ibis.common as com

from ibis.pandas.dispatch import execute


def compute_sort_key(key, data, **kwargs):
    by = key.args[0]
    try:
        return by.get_name(), None
    except com.ExpressionError:
        name = ibis.util.guid()
        new_scope = {t: data for t in by.op().root_tables()}
        new_column = execute(by, new_scope, **kwargs)
        new_column.name = name
        return name, new_column


def compute_sorted_frame(sort_keys, df, **kwargs):
    computed_sort_keys = []
    ascending = [key.op().ascending for key in sort_keys]
    new_columns = {}

    for i, key in enumerate(map(operator.methodcaller('op'), sort_keys)):
        computed_sort_key, temporary_column = compute_sort_key(
            key, df, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)

        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(computed_sort_keys, ascending=ascending)
    result = result.drop(new_columns.keys(), axis=1)
    return result
