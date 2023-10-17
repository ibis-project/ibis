"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping
import toolz

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.exceptions import UnsupportedOperationError
from ibis.common.patterns import pattern, replace
from ibis.common.deferred import var
from ibis.common.patterns import Check, Eq, NoMatch, Pattern, pattern, replace, In
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
x = var("x")
y = var("y")
t = var("t")
r = var("r")
parent = var("parent")
fields = var("fields")
parent_fields = var("parent_fields")

# JOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# reprojection of the same fields from the parent selection
rewrite_redundant_selection = (
    p.Selection(x, selections=[p.Selection(selections=Eq(x.selections))]) >> x
)


# the following rewrite rule is responsible to fuse the following two non-overlapping
# selections into a single one:
#
# INPUT:
#
# r0 := UnboundTable: t
#   col int32
#
# r1 := Selection[r0]
#   selections:
#     r0
#     col1: r0.col + 1
#
# Selection[r1]
#   selections:
#     r1
#     col2: r0.col + 2
#
# OUTPUT:
#
# r0 := UnboundTable: t
#   col int32

# Selection[r0]
#   selections:
#     r0
#     col1: r0.col + 1
#     col2: r0.col + 2


def can_prune_projection(projection, context):
    parent = context["parent"]
    fields = context["fields"]
    parent_fields = context["parent_fields"]

    projected_column_names = []
    for value in parent_fields:
        if isinstance(value, (ops.Relation, ops.TableColumn)):
            # we are only interested in projected value expressions, not tables
            # nor column references which are not changing the projection
            continue
        elif value.find((ops.WindowFunction, ops.ExistsSubquery), filter=ops.Value):
            # the parent has analytic projections like window functions so we
            # can't push down filters to that level
            return NoMatch
        else:
            # otherwise collect the names of newly projected value expressions
            # which are not just plain column references
            projected_column_names.append(value.name)

    conflicting_table_columns = p.TableColumn(parent, In(projected_column_names))
    for value in fields:
        if value.match(conflicting_table_columns, filter=ops.Value):
            return NoMatch

    return projection


@replace(
    p.Selection(parent @ p.Selection(selections=parent_fields), selections=fields)
    & can_prune_projection
)
def prune_subsequent_projection(_, parent, fields, parent_fields):
    print(_.to_expr())

    pattern = Eq(parent) >> parent.table

    selections = []
    for field in fields:
        if field == parent:
            selections.extend(parent_fields)
        else:
            selections.append(field.replace(pattern))

    return parent.copy(selections=selections)
