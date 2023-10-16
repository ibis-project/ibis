"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.exceptions import UnsupportedOperationError
from ibis.common.patterns import pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)


@replace(p.FillNa)
def rewrite_fillna(_):
    """Rewrite FillNa expressions to use more common operations."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.table.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.table

    selections = []
    for name in _.table.schema.names:
        col = ops.TableColumn(_.table, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Alias(ops.Coalesce((col, value)), name)
        selections.append(col)

    return ops.Selection(_.table, selections, (), ())


@replace(p.DropNa)
def rewrite_dropna(_):
    """Rewrite DropNa expressions to use more common operations."""
    if _.subset is None:
        columns = [ops.TableColumn(_.table, name) for name in _.table.schema.names]
    else:
        columns = _.subset

    if columns:
        preds = [
            functools.reduce(
                ops.And if _.how == "any" else ops.Or,
                [ops.NotNull(c) for c in columns],
            )
        ]
    elif _.how == "all":
        preds = [ops.Literal(False, dtype=dt.bool)]
    else:
        return _.table

    return ops.Selection(_.table, (), preds, ())


@replace(p.Sample)
def rewrite_sample(_):
    """Rewrite Sample as `t.filter(random() <= fraction)`.

    Errors as unsupported if a `seed` is specified.
    """

    if _.seed is not None:
        raise UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )

    return ops.Selection(
        _.table,
        (),
        (ops.LessEqual(ops.RandomScalar(), _.fraction),),
        (),
    )
