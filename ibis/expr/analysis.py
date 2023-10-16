from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import toolz

import ibis.common.graph as g
import ibis.expr.operations as ops
import ibis.expr.operations.relations as rels
import ibis.expr.types as ir
from ibis import util
from ibis.common.deferred import _, deferred, var
from ibis.common.exceptions import IbisTypeError, IntegrityError
from ibis.common.patterns import Eq, In, pattern
from ibis.util import Namespace

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

p = Namespace(pattern, module=ops)
c = Namespace(deferred, module=ops)

x = var("x")
y = var("y")

# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_immediate_parents(node: ops.Node, table: ops.TableNode) -> ops.Node:
    """Replace immediate parent tables in `op` with `table`."""
    parents = find_immediate_parent_tables(node)
    return node.replace(In(parents) >> table)


def find_immediate_parent_tables(input_node, keep_input=True):
    """Find every first occurrence of a `ir.Table` object in `input_node`.

    This function does not traverse into `Table` objects. For example, the
    underlying `PhysicalTable` of a `Selection` will not be yielded.

    Parameters
    ----------
    input_node
        Input node
    keep_input
        Whether to keep the input when traversing

    Yields
    ------
    ir.Expr
        Parent table expression

    Examples
    --------
    >>> import ibis, toolz
    >>> t = ibis.table([("a", "int64")], name="t")
    >>> expr = t.mutate(foo=t.a + 1)
    >>> (result,) = find_immediate_parent_tables(expr.op())
    >>> result.equals(expr.op())
    True
    >>> (result,) = find_immediate_parent_tables(expr.op(), keep_input=False)
    >>> result.equals(t.op())
    True
    """
    assert all(isinstance(arg, ops.Node) for arg in util.promote_list(input_node))

    def finder(node):
        if isinstance(node, ops.TableNode):
            if keep_input or node != input_node:
                return g.halt, node
            else:
                return g.proceed, None

        # HACK: special case ops.Contains to only consider the needle's base
        # table, since that's the only expression that matters for determining
        # cardinality
        elif isinstance(node, ops.InColumn):
            # we allow InColumn.options to be a column from a foreign table
            return [node.value], None
        else:
            return g.proceed, None

    return list(toolz.unique(g.traverse(finder, input_node)))


def get_mutation_exprs(exprs: list[ir.Expr], table: ir.Table) -> list[ir.Expr | None]:
    """Return the exprs to use to instantiate the mutation."""
    # The below logic computes the mutation node exprs by splitting the
    # assignment exprs into two disjoint sets:
    # 1) overwriting_cols_to_expr, which maps a column name to its expr
    # if the expr contains a column that overwrites an existing table column.
    # All keys in this dict are columns in the original table that are being
    # overwritten by an assignment expr.
    # 2) non_overwriting_exprs, which is a list of all exprs that do not do
    # any overwriting. That is, if an expr is in this list, then its column
    # name does not exist in the original table.
    # Given these two data structures, we can compute the mutation node exprs
    # based on whether any columns are being overwritten.
    overwriting_cols_to_expr: dict[str, ir.Expr | None] = {}
    non_overwriting_exprs: list[ir.Expr] = []
    table_schema = table.schema()
    for expr in exprs:
        expr_contains_overwrite = False
        if isinstance(expr, ir.Value) and expr.get_name() in table_schema:
            overwriting_cols_to_expr[expr.get_name()] = expr
            expr_contains_overwrite = True

        if not expr_contains_overwrite:
            non_overwriting_exprs.append(expr)

    columns = table.columns
    if overwriting_cols_to_expr:
        return [
            overwriting_cols_to_expr.get(column, table[column])
            for column in columns
            if overwriting_cols_to_expr.get(column, table[column]) is not None
        ] + non_overwriting_exprs

    table_expr: ir.Expr = table
    return [table_expr] + exprs


def pushdown_selection_filters(parent, predicates):
    if not predicates:
        return parent

    default = ops.Selection(parent, selections=[], predicates=predicates)
    if not isinstance(parent, (ops.Selection, ops.Aggregation)):
        return default

    projected_column_names = set()
    for value in parent._projection.selections:
        if isinstance(value, (ops.Relation, ops.TableColumn)):
            # we are only interested in projected value expressions, not tables
            # nor column references which are not changing the projection
            continue
        elif value.find((ops.WindowFunction, ops.ExistsSubquery), filter=ops.Value):
            # the parent has analytic projections like window functions so we
            # can't push down filters to that level
            return default
        else:
            # otherwise collect the names of newly projected value expressions
            # which are not just plain column references
            projected_column_names.add(value.name)

    conflicting_projection = p.TableColumn(parent, In(projected_column_names))
    pushdown_pattern = Eq(parent) >> parent.table

    simplified = []
    for pred in predicates:
        if pred.match(conflicting_projection, filter=p.Value):
            return default
        try:
            simplified.append(pred.replace(pushdown_pattern))
        except (IntegrityError, IbisTypeError):
            # former happens when there is a duplicate column name in the parent
            # which is a join, the latter happens for semi/anti joins
            return default

    return parent.copy(predicates=parent.predicates + tuple(simplified))


def windowize_function(expr, default_frame, merge_frames=False):
    func, frame = var("func"), var("frame")

    wrap_analytic = (p.Analytic | p.Reduction) >> c.WindowFunction(_, default_frame)
    merge_windows = p.WindowFunction(func, frame) >> c.WindowFunction(
        func,
        frame.copy(
            order_by=frame.order_by + default_frame.order_by,
            group_by=frame.group_by + default_frame.group_by,
        ),
    )

    node = expr.op()
    if merge_frames:
        # it only happens in ibis.expr.groupby.GroupedTable, but the projector
        # changes the windowization order to put everything here
        node = node.replace(merge_windows, filter=p.Value)
    node = node.replace(wrap_analytic, filter=p.Value & ~p.WindowFunction)

    return node.to_expr()


def contains_first_or_last_agg(exprs):
    def fn(node: ops.Node) -> tuple[bool, bool | None]:
        if not isinstance(node, ops.Value):
            return g.halt, None
        return g.proceed, isinstance(node, (ops.First, ops.Last))

    return any(g.traverse(fn, exprs))


def simplify_aggregation(agg):
    def _pushdown(nodes):
        subbed = []
        for node in nodes:
            new_node = node.replace(Eq(agg.table) >> agg.table.table)
            subbed.append(new_node)

        # TODO(kszucs): perhaps this validation could be omitted
        if subbed:
            valid = shares_all_roots(subbed, agg.table.table)
        else:
            valid = True

        return valid, subbed

    table = agg.table
    if (
        isinstance(table, ops.Selection)
        and not table.selections
        # more aggressive than necessary, a better solution would be to check
        # whether the selections have any order sensitive aggregates that
        # *depend on* the sort_keys
        and not (table.sort_keys or contains_first_or_last_agg(table.selections))
    ):
        metrics_valid, lowered_metrics = _pushdown(agg.metrics)
        by_valid, lowered_by = _pushdown(agg.by)
        having_valid, lowered_having = _pushdown(agg.having)

        if metrics_valid and by_valid and having_valid:
            valid_lowered_sort_keys = frozenset(lowered_metrics).union(lowered_by)
            return ops.Aggregation(
                table.table,
                lowered_metrics,
                by=lowered_by,
                having=lowered_having,
                predicates=agg.table.predicates,
                # only the sort keys that exist as grouping keys or metrics can
                # be included
                sort_keys=[
                    key
                    for key in agg.table.sort_keys
                    if key.expr in valid_lowered_sort_keys
                ],
            )

    return agg


class Projector:
    """Analysis and validation of projection operation.

    This pass tries to take advantage of projection fusion opportunities where
    they exist, i.e. combining compatible projections together rather than
    nesting them.

    Translation / evaluation later will not attempt to do any further fusion /
    simplification.
    """

    def __init__(self, parent, proj_exprs):
        # TODO(kszucs): rewrite projector to work with operations exclusively
        proj_exprs = util.promote_list(proj_exprs)
        self.parent = parent
        self.input_exprs = proj_exprs
        self.resolved_exprs = [parent._ensure_expr(e) for e in proj_exprs]

        default_frame = ops.RowsWindowFrame(table=parent)
        self.clean_exprs = [
            windowize_function(expr, default_frame) for expr in self.resolved_exprs
        ]

    def get_result(self):
        roots = find_immediate_parent_tables(self.parent.op())
        first_root = roots[0]
        parent_op = self.parent.op()

        # reprojection of the same selections
        if len(self.clean_exprs) == 1:
            first = self.clean_exprs[0].op()
            if isinstance(first, ops.Selection):
                if first.selections == parent_op.selections:
                    return parent_op

        if len(roots) == 1 and isinstance(first_root, ops.Selection):
            fused_op = self.try_fusion(first_root)
            if fused_op is not None:
                return fused_op

        return ops.Selection(self.parent, self.clean_exprs)

    def try_fusion(self, root):
        assert self.parent.op() == root

        root_table = root.table
        root_table_expr = root_table.to_expr()
        roots = find_immediate_parent_tables(root_table)
        fused_exprs = []
        clean_exprs = self.clean_exprs

        if not isinstance(root_table, ops.Join):
            try:
                resolved = [
                    root_table_expr._ensure_expr(expr) for expr in self.input_exprs
                ]
            except (AttributeError, IbisTypeError):
                resolved = clean_exprs
            else:
                # if any expressions aren't exactly equivalent then don't try
                # to fuse them
                if any(
                    not res_root_root.equals(res_root)
                    for res_root_root, res_root in zip(resolved, clean_exprs)
                ):
                    return None
        else:
            # joins cannot be used to resolve expressions, but we still may be
            # able to fuse columns from a projection off of a join. In that
            # case, use the projection's input expressions as the columns with
            # which to attempt fusion
            resolved = clean_exprs

        root_selections = root.selections
        parent_op = self.parent.op()
        for val in resolved:
            # a * projection
            if isinstance(val, ir.Table) and (
                parent_op.equals(val.op())
                # gross we share the same table root. Better way to
                # detect?
                or len(roots) == 1
                and find_immediate_parent_tables(val.op())[0] == roots[0]
            ):
                have_root = False
                for root_sel in root_selections:
                    # Don't add the * projection twice
                    if root_sel.equals(root_table):
                        fused_exprs.append(root_table)
                        have_root = True
                        continue
                    fused_exprs.append(root_sel)

                # This was a filter, so implicitly a select *
                if not have_root and not root_selections:
                    fused_exprs = [root_table, *fused_exprs]
            elif shares_all_roots(val.op(), root_table):
                fused_exprs.append(val)
            else:
                return None

        return ops.Selection(
            root_table,
            fused_exprs,
            predicates=root.predicates,
            sort_keys=root.sort_keys,
        )


def find_first_base_table(node):
    def predicate(node):
        if isinstance(node, ops.TableNode):
            return g.halt, node
        else:
            return g.proceed, None

    try:
        return next(g.traverse(predicate, node))
    except StopIteration:
        return None


def _find_projections(node):
    if isinstance(node, ops.Selection):
        # remove predicates and sort_keys, so that child tables are considered
        # equivalent even if their predicates and sort_keys are not
        return g.proceed, node._projection
    elif isinstance(node, ops.SelfReference):
        return g.proceed, node
    elif isinstance(node, ops.Aggregation):
        return g.proceed, node._projection
    elif isinstance(node, ops.Join):
        return g.proceed, None
    elif isinstance(node, ops.TableNode):
        return g.halt, node
    elif isinstance(node, ops.InColumn):
        # we allow InColumn.options to be a column from a foreign table
        return [node.value], None
    else:
        return g.proceed, None


def shares_all_roots(exprs, parents):
    # unique table dependencies of exprs and parents
    exprs_deps = set(g.traverse(_find_projections, exprs))
    parents_deps = set(g.traverse(_find_projections, parents))
    return exprs_deps <= parents_deps


def shares_some_roots(exprs, parents):
    # unique table dependencies of exprs and parents
    exprs_deps = set(g.traverse(_find_projections, exprs))
    parents_deps = set(g.traverse(_find_projections, parents))
    # Also return True if exprs has no roots (e.g. literal-only expressions)
    return bool(exprs_deps & parents_deps) or not exprs_deps


def flatten_predicate(node):
    """Yield the expressions corresponding to the `And` nodes of a predicate.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
    >>> filt = (t.a == 1) & (t.b == "foo")
    >>> predicates = flatten_predicate(filt.op())
    >>> len(predicates)
    2
    >>> predicates[0].to_expr().name("left")
    r0 := UnboundTable: t
      a int64
      b string
    left: r0.a == 1
    >>> predicates[1].to_expr().name("right")
    r0 := UnboundTable: t
      a int64
      b string
    right: r0.b == 'foo'
    """

    def predicate(node):
        if isinstance(node, ops.And):
            return g.proceed, None
        else:
            return g.halt, node

    return list(g.traverse(predicate, node))


def find_predicates(node, flatten=True):
    # TODO(kszucs): consider to remove flatten argument and compose with
    # flatten_predicates instead
    def predicate(node):
        assert isinstance(node, ops.Node), type(node)
        if isinstance(node, ops.Value) and node.dtype.is_boolean():
            if flatten and isinstance(node, ops.And):
                return g.proceed, None
            else:
                return g.halt, node
        return g.proceed, None

    return list(g.traverse(predicate, node))


def find_subqueries(node: ops.Node, min_dependents=1) -> tuple[ops.Node, ...]:
    subquery_dependents = defaultdict(set)
    for n in filter(None, util.promote_list(node)):
        dependents = g.Graph.from_dfs(n).invert()
        for u, vs in dependents.toposort().items():
            # count the number of table-node dependents on the current node
            # but only if the current node is a selection or aggregation
            if isinstance(u, (rels.Projection, rels.Aggregation, rels.Limit)):
                subquery_dependents[u].update(vs)

    return tuple(
        node
        for node, dependents in reversed(subquery_dependents.items())
        if len(dependents) >= min_dependents
    )


def find_toplevel_unnest_children(nodes: Iterable[ops.Node]) -> Iterator[ops.Table]:
    def finder(node):
        return (
            isinstance(node, ops.Value),
            find_first_base_table(node) if isinstance(node, ops.Unnest) else None,
        )

    return g.traverse(finder, nodes, filter=ops.Node)
