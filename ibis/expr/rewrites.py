"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import UnsupportedOperationError
from ibis.common.patterns import Check, pattern, replace
from ibis.util import Namespace, gen_name

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


y = var("y")
name = var("name")


# TODO(kszucs): must be updated
@replace(p.FillNa)
def rewrite_fillna(_):
    """Rewrite FillNa expressions to use more common operations."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.parent.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.parent

    selections = []
    for name in _.parent.schema.names:
        col = ops.TableColumn(_.parent, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Alias(ops.Coalesce((col, value)), name)
        selections.append(col)

    return ops.Project(_.parent, selections)


# TODO(kszucs): must be updated
@replace(p.DropNa)
def rewrite_dropna(_):
    """Rewrite DropNa expressions to use more common operations."""
    if _.subset is None:
        columns = [ops.TableColumn(_.parent, name) for name in _.parent.schema.names]
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
        return _.parent

    return ops.Filter(_.parent, tuple(preds))


# TODO(kszucs): must be updated
@replace(p.Sample)
def rewrite_sample(_):
    """Rewrite Sample as `t.filter(random() <= fraction)`.

    Errors as unsupported if a `seed` is specified.
    """

    if _.seed is not None:
        raise UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )

    return ops.Filter(_.parent, (ops.LessEqual(ops.RandomScalar(), _.fraction),))


@replace(ops.Analytic)
def project_wrap_analytic(_, rel):
    # Wrap analytic functions in a window function
    return ops.WindowFunction(_, ops.RowsWindowFrame(rel))


@replace(ops.Reduction)
def project_wrap_reduction(_, rel):
    # Query all the tables that the reduction depends on
    if _.relations == {rel}:
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
        return ops.ScalarSubquery(_.to_expr().as_table())


ReductionValue = p.Reduction | p.Field(p.Aggregate(groups={}))


@replace(ReductionValue)
def filter_wrap_reduction_value(_):
    if isinstance(_, ops.Field):
        value = _.rel.fields[_.name]
    else:
        value = _
    return ops.ScalarSubquery(value.to_expr().as_table())


# TODO(kszucs): schema comparison should be updated to not distinguish between
# different column order
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


@replace(ops.Aggregate)
def aggregate_to_groupby(_):
    if not _.groups:
        return ops.PandasProject(_.parent, _.metrics)

    values = {}

    for v in _.groups.values():
        if not isinstance(v, ops.Field):
            values[v] = gen_name("agg")

    for v in _.metrics.values():
        for red in v.find_topmost(ops.Reduction):
            for arg in red.args:
                if isinstance(arg, ops.Value) and not isinstance(arg, ops.Field):
                    values[arg] = gen_name("agg")

    fields = {k: ops.Field(_.parent, k) for k in _.parent.schema}
    fields.update({v: k for k, v in values.items()})
    proj = ops.Project(_.parent, fields)

    mapping = {v: k for k, v in proj.fields.items()}
    groups = [v.replace(mapping, filter=ops.Value) for k, v in _.groups.items()]
    groupby = ops.GroupBy(proj, groups)

    # turn these into a different type, e.g. LazyField to not compute it
    mapping = {v: ops.Field(groupby, k) for k, v in proj.fields.items()}
    metrics = {k: v.replace(mapping) for k, v in _.metrics.items()}
    return ops.GroupByMetrics(groupby, metrics)
