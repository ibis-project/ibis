"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import UnsupportedOperationError
from ibis.common.patterns import Check, pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


y = var("y")
name = var("name")


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


@replace(ops.Analytic)
def project_wrap_analytic(_, rel):
    # Wrap analytic functions in a window function
    return ops.WindowFunction(_, ops.RowsWindowFrame(rel))


@replace(ops.Reduction)
def project_wrap_reduction(_, rel):
    # Query all the tables that the reduction depends on
    parents = _.find_topmost(ops.Relation)  # need to add Subquery here?
    if parents == [rel]:
        # The reduction is fully originating from the `rel`, so turn
        # it into a window function of `rel`
        return ops.WindowFunction(_, ops.RowsWindowFrame(rel))
    else:
        # 1. The reduction doesn't depend on any table, constructed from
        #    scalar values, so turn it into a scalar subquery.
        # 2. The reduction is originating from `rel` and other tables,
        #    so this is a correlated scalar subquery.
        # 3. The reduction is originating entirely from other tables,
        #    so this is an uncorrelated scalar subquery.
        return ops.ScalarSubquery(_)


@replace(ops.Reduction)
def filter_wrap_reduction(_):
    return ops.ScalarSubquery(_)


@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != ops.Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return ops.Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule) for v in _.predicates)
    return ops.Filter(y.parent, y.predicates + preds)


@replace(p.Filter(y @ p.Project))
def reorder_filter_project(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    preds = tuple(v.replace(rule) for v in _.predicates)

    inner = ops.Filter(y.parent, preds)
    rule = p.Field(y.parent, name) >> d.Field(inner, name)
    projs = {k: v.replace(rule) for k, v in y.values.items()}

    return ops.Project(inner, projs)


# TODO(kszucs): rewrite projection not actually depending on the parent table,
# it should rather use a DummyTable object, see the following repr:
# r0 := UnboundTable: t
#   bool_col   boolean
#   int_col    int64
#   float_col  float64
#   string_col string
#
# Project[r0]
#   res: Subquery(Sum(Unnest((1, 2, 3))))

# TODO(kszucs): add a rewrite rule for nestes JoinChain objects where the
# JoinLink depends on another JoinChain, in this case the JoinLink should be
# merged into the JoinChain


# TODO(kszucs): this may work if the sort keys are not overlapping, need to revisit
# @replace(p.Sort(y @ p.Sort))
# def subsequent_sorts(_, y):
#     return Sort(y.parent, y.keys + _.keys)

# TODO(kszucs): should use node.map() instead of node.replace() to match on the
# largest possible pattern, or even better `graph.traverse()` to go top-down
# possibly need a top-to-bottom rewriter for rules like the one below


# replacement rule to convert a sequence of project filter operations into a
# SQL-like ops.Selection operation
@replace(
    "sort" @ p.Sort("proj" @ p.Project("filt" @ p.Filter("root" @ p.Relation)))
    | "proj" @ p.Project("filt" @ p.Filter("root" @ p.Relation))
    | "sort" @ p.Sort("root" @ p.Relation)
    | "filt" @ p.Filter("root" @ p.Relation)
    | "proj" @ p.Project("root" @ p.Relation)
)
def sequalize(_, root, filt=None, proj=None, sort=None):
    selections = proj.values if proj else {}
    predicates = filt.predicates if filt else ()
    sort_keys = sort.keys if sort else ()
    parent = root

    if filt:
        rule = p.Field(filt, name) >> d.Field(root, name)
        selections = {k: v.replace(rule) for k, v in selections.items()}

    if proj:
        rule = p.Field(proj, name) >> Item(selections, name)
        sort_keys = tuple(v.replace(rule) for v in sort_keys)

    return ops.Selection(
        parent=parent, selections=selections, predicates=predicates, sort_keys=sort_keys
    )
