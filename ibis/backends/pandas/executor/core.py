from __future__ import annotations

from functools import singledispatch

import ibis.expr.operations as ops
from ibis.common.exceptions import OperationNotDefinedError
from ibis.formats.pandas import PandasData


@singledispatch
def execute(node, **kwargs):
    raise OperationNotDefinedError(f"no rule for {type(node)}")


def zuper(node, params):
    from ibis.backends.pandas.rewrites import (
        rewrite_aggregate,
        rewrite_join,
        rewrite_project,
    )
    from ibis.expr.rewrites import p

    replace_literals = p.ScalarParameter >> (
        lambda _: ops.Literal(value=params[_], dtype=_.dtype)
    )

    def fn(node, _, **kwargs):
        # TODO(kszucs): need to clean up the resultset as soon as an intermediate
        # result is not needed anymore
        result = execute(node, **kwargs)
        return result

    original = node

    node = node.to_expr().as_table().op()
    node = node.replace(
        rewrite_project | rewrite_aggregate | rewrite_join | replace_literals
    )
    # print(node.to_expr())
    df = node.map(fn)[node]

    # TODO(kszucs): add a flag to disable this conversion because it can be
    # expensive for columns with object dtype
    df = PandasData.convert_table(df, node.schema)
    if isinstance(original, ops.Value):
        if original.shape.is_scalar():
            return df.iloc[0, 0]
        elif original.shape.is_columnar():
            return df.iloc[:, 0]
        else:
            raise TypeError(f"Unexpected shape: {original.shape}")
    else:
        return df
