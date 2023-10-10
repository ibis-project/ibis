from __future__ import annotations

import functools
import operator
from collections import defaultdict
from typing import TYPE_CHECKING

import toolz

import ibis.common.graph as g
import ibis.expr.operations as ops
import ibis.expr.operations.relations as rels
import ibis.expr.types as ir
from ibis import util
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import IbisTypeError, IntegrityError
from ibis.common.patterns import Call, Object, Variable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

p = Object.namespace(ops)
c = Call.namespace(ops)

x = Variable("x")
y = Variable("y")

# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_for(node: ops.Node, substitutions: Mapping[ops.Node, ops.Node]) -> ops.Node:
    """Substitute operations in `node` with nodes in `substitutions`.

    Parameters
    ----------
    node
        An Ibis operation
    substitutions
        A mapping from node to node. If any subnode of `node` is equal to any
        of the keys in `substitutions`, the value for that key will replace the
        corresponding node in `node`.

    Returns
    -------
    Node
        An Ibis expression
    """
    assert isinstance(node, ops.Node), type(node)

    def fn(node):
        try:
            return substitutions[node]
        except KeyError:
            if isinstance(node, ops.TableNode):
                return g.halt
            return g.proceed

    return substitute(fn, node)


def sub_immediate_parents(op: ops.Node, table: ops.TableNode) -> ops.Node:
    """Replace immediate parent tables in `op` with `table`."""
    return sub_for(op, {base: table for base in find_immediate_parent_tables(op)})


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


def substitute(fn, node):
    """Substitute expressions with other expressions."""

    assert isinstance(node, ops.Node), type(node)

    result = fn(node)
    if result is g.halt:
        return node
    elif result is not g.proceed:
        assert isinstance(result, ops.Node), type(result)
        return result

    new_args = []
    for arg in node.args:
        if isinstance(arg, tuple):
            arg = tuple(
                substitute(fn, x) if isinstance(arg, ops.Node) else x for x in arg
            )
        elif isinstance(arg, ops.Node):
            arg = substitute(fn, arg)
        new_args.append(arg)

    try:
        return node.__class__(*new_args)
    except (TypeError, ValidationError):
        return node


def substitute_parents(node):
    """Rewrite `node` by replacing table nodes that commute."""
    assert isinstance(node, ops.Node), type(node)

    def fn(node):
        if isinstance(node, ops.Selection):
            # stop substituting child nodes
            return g.halt
        elif isinstance(node, ops.TableColumn):
            # For table column references, in the event that we're on top of a
            # projection, we need to check whether the ref comes from the base
            # table schema or is a derived field. If we've projected out of
            # something other than a physical table, then lifting should not
            # occur
            table = node.table

            if isinstance(table, ops.Selection):
                for val in table.selections:
                    if isinstance(val, ops.PhysicalTable) and node.name in val.schema:
                        return ops.TableColumn(val, node.name)

        # keep looking for nodes to substitute
        return g.proceed

    return substitute(fn, node)


def substitute_unbound(node):
    """Rewrite `node` by replacing table expressions with an equivalent unbound table."""
    return node.replace(
        p.DatabaseTable(name=x, schema=y) >> c.UnboundTable(name=x, schema=y)
    )


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


def apply_filter(op, predicates):
    # This will attempt predicate pushdown in the cases where we can do it
    # easily and safely, to make both cleaner SQL and fewer referential errors
    # for users
    if not predicates:
        return op

    if isinstance(op, ops.Selection):
        return pushdown_selection_filters(op, predicates)
    elif isinstance(op, ops.Aggregation):
        return pushdown_aggregation_filters(op, predicates)
    else:
        return ops.Selection(op, [], predicates)


def pushdown_selection_filters(op, predicates):
    default = ops.Selection(op, selections=[], predicates=predicates)

    # We can't push down filters on Unnest or Window because they
    # change the shape and potential values of the data.
    if any(
        isinstance(
            sel.arg if isinstance(sel, ops.Alias) else sel,
            (ops.Unnest, ops.Window),
        )
        for sel in op.selections
    ):
        return default

    # if any of the filter predicates have the parent expression among
    # their roots, then pushdown (at least of that predicate) is not
    # possible

    # It's not unusual for the filter to reference the projection
    # itself. If a predicate can be pushed down, in this case we must
    # rewrite replacing the table refs with the roots internal to the
    # projection we are referencing
    #
    # Assuming that the fields referenced by the filter predicate originate
    # below the projection, we need to rewrite the predicate referencing
    # the parent tables in the join being projected

    # Potential fusion opportunity. The predicates may need to be
    # rewritten in terms of the child table. This prevents the broken
    # ref issue (described in more detail in #59)
    try:
        simplified_predicates = tuple(
            sub_for(predicate, {op: op.table})
            if not is_reduction(predicate)
            else predicate
            for predicate in predicates
        )
    except IntegrityError:
        return default

    if not shares_all_roots(simplified_predicates, op.table):
        return default

    # find spuriously simplified predicates
    for predicate in simplified_predicates:
        # find columns in the predicate
        depends_on = predicate.find((ops.TableColumn, ops.Literal))
        for projection in op.selections:
            if not isinstance(projection, (ops.TableColumn, ops.Literal)):
                # if the projection's table columns overlap with columns
                # used in the predicate then we return immediately
                #
                # this means that we were too aggressive during simplification
                # example: t.mutate(a=_.a + 1).filter(_.a > 1)
                if projection.find((ops.TableColumn, ops.Literal)) & depends_on:
                    return default

    return ops.Selection(
        op.table,
        selections=op.selections,
        predicates=op.predicates + simplified_predicates,
        sort_keys=op.sort_keys,
    )


def pushdown_aggregation_filters(op, predicates):
    # Potential fusion opportunity
    # GH1344: We can't sub in things with correlated subqueries
    simplified_predicates = tuple(
        # Originally this line tried substituting op.table in for expr, but
        # that is too aggressive in the presence of filters that occur
        # after aggregations.
        #
        # See https://github.com/ibis-project/ibis/pull/3341 for details
        sub_for(predicate, {op.table: op}) if not is_reduction(predicate) else predicate
        for predicate in predicates
    )

    if shares_all_roots(simplified_predicates, op.table):
        return ops.Aggregation(
            op.table,
            op.metrics,
            by=op.by,
            having=op.having,
            predicates=op.predicates + simplified_predicates,
            sort_keys=op.sort_keys,
        )
    else:
        return ops.Selection(op, [], predicates)


# TODO(kszucs): use ibis.expr.analysis.substitute instead
def propagate_down_window(func: ops.Value, frame: ops.WindowFrame):
    import ibis.expr.operations as ops

    clean_args = []
    for arg in func.args:
        if isinstance(arg, ops.Value) and not isinstance(func, ops.WindowFunction):
            arg = propagate_down_window(arg, frame)
            if isinstance(arg, ops.Analytic):
                arg = ops.WindowFunction(arg, frame)
        clean_args.append(arg)

    return type(func)(*clean_args)


# TODO(kszucs): rewrite to receive and return an ops.Node
def windowize_function(expr, frame):
    assert isinstance(expr, ir.Expr), type(expr)
    assert isinstance(frame, ops.WindowFrame)

    def _windowize(op, frame):
        if isinstance(op, ops.WindowFunction):
            walked_child = _walk(op.func, frame)
            walked = walked_child.to_expr().over(op.frame).op()
        elif isinstance(op, ops.Value):
            walked = _walk(op, frame)
        else:
            walked = op

        if isinstance(walked, (ops.Analytic, ops.Reduction)):
            return op.to_expr().over(frame).op()
        elif isinstance(walked, ops.WindowFunction):
            if frame is not None:
                frame = walked.frame.copy(
                    group_by=frame.group_by + walked.frame.group_by,
                    order_by=frame.order_by + walked.frame.order_by,
                )
                return walked.to_expr().over(frame).op()
            else:
                return walked
        else:
            return walked

    def _walk(op, frame):
        # TODO(kszucs): rewrite to use the substitute utility
        windowed_args = []
        for arg in op.args:
            if isinstance(arg, ops.Value):
                arg = _windowize(arg, frame)
            elif isinstance(arg, tuple):
                arg = tuple(_windowize(x, frame) for x in arg)

            windowed_args.append(arg)

        return type(op)(*windowed_args)

    return _windowize(expr.op(), frame).to_expr()


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
            subbed.append(sub_for(node, {agg.table: agg.table.table}))

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
            windowize_function(expr, frame=default_frame)
            for expr in self.resolved_exprs
        ]

    def get_result(self):
        roots = find_immediate_parent_tables(self.parent.op())
        first_root = roots[0]

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


def is_reduction(node):
    """Check whether an expression contains a reduction or not.

    Aggregations yield typed scalar expressions, since the result of an
    aggregation is a single value. When creating an table expression
    containing a GROUP BY equivalent, we need to be able to easily check
    that we are looking at the result of an aggregation.

    As an example, the expression we are looking at might be something
    like: foo.sum().log10() + bar.sum().log10()

    We examine the operator DAG in the expression to determine if there
    are aggregations present.

    A bound aggregation referencing a separate table is a "false
    aggregation" in a GROUP BY-type expression and should be treated a
    literal, and must be computed as a separate query and stored in a
    temporary variable (or joined, for bound aggregations with keys)
    """

    def predicate(node):
        if isinstance(node, ops.Reduction):
            return g.halt, True
        elif isinstance(node, ops.TableNode):
            # don't go below any table nodes
            return g.halt, None
        else:
            return g.proceed, None

    return any(g.traverse(predicate, node))


_ANY_OP_MAPPING = {
    ops.Any: ops.UnresolvedExistsSubquery,
    ops.NotAny: ops.UnresolvedNotExistsSubquery,
}


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


# TODO(kszucs): move to types/logical.py
def _make_any(
    expr,
    any_op_class: type[ops.Any] | type[ops.NotAny],
    *,
    where: ir.BooleanValue | None = None,
):
    assert isinstance(expr, ir.Expr), type(expr)

    tables = find_immediate_parent_tables(expr.op())
    predicates = find_predicates(expr.op(), flatten=True)

    if len(tables) > 1:
        op = _ANY_OP_MAPPING[any_op_class](
            tables=[t.to_expr() for t in tables],
            predicates=predicates,
        )
    else:
        op = any_op_class(expr, where=expr._bind_reduction_filter(where))
    return op.to_expr()


# TODO(kszucs): use substitute instead
@functools.singledispatch
def _rewrite_filter(op, **kwargs):
    raise NotImplementedError(type(op))


@_rewrite_filter.register(ops.Reduction)
def _rewrite_filter_reduction(op, name: str | None = None, **kwargs):
    """Turn a reduction inside of a filter into an aggregate."""
    # TODO: what about reductions that reference a join that isn't visible at
    # this level? Means we probably have the wrong design, but will have to
    # revisit when it becomes a problem.
    if name is not None:
        op = ops.Alias(op, name=name)

    agg = op.to_expr().as_table()
    return ops.TableArrayView(agg)


@_rewrite_filter.register(ops.Any)
@_rewrite_filter.register(ops.TableColumn)
@_rewrite_filter.register(ops.Literal)
@_rewrite_filter.register(ops.ExistsSubquery)
@_rewrite_filter.register(ops.NotExistsSubquery)
@_rewrite_filter.register(ops.WindowFunction)
def _rewrite_filter_subqueries(op, **kwargs):
    """Don't rewrite any of these operations in filters."""
    return op


@_rewrite_filter.register(ops.Alias)
def _rewrite_filter_alias(op, name: str | None = None, **kwargs):
    """Rewrite filters on aliases."""
    return _rewrite_filter(
        op.arg,
        name=name if name is not None else op.name,
        **kwargs,
    )


@_rewrite_filter.register(ops.Value)
def _rewrite_filter_value(op, **kwargs):
    """Recursively apply filter rewriting on operations."""

    visited = [
        _rewrite_filter(arg, **kwargs) if isinstance(arg, ops.Node) else arg
        for arg in op.args
    ]
    if all(map(operator.is_, visited, op.args)):
        return op

    return op.__class__(*visited)


@_rewrite_filter.register(tuple)
def _rewrite_filter_value_list(op, **kwargs):
    visited = [
        _rewrite_filter(arg, **kwargs) if isinstance(arg, ops.Node) else arg
        for arg in op.args
    ]

    if all(map(operator.is_, visited, op.args)):
        return op

    return op.__class__(*visited)


def find_toplevel_unnest_children(nodes: Iterable[ops.Node]) -> Iterator[ops.Table]:
    def finder(node):
        return (
            isinstance(node, ops.Value),
            find_first_base_table(node) if isinstance(node, ops.Unnest) else None,
        )

    return g.traverse(finder, nodes, filter=ops.Node)


def find_toplevel_aggs(nodes: Iterable[ops.Node]) -> Iterator[ops.Table]:
    def finder(node):
        return (
            isinstance(node, ops.Value),
            node if isinstance(node, ops.Reduction) else None,
        )

    return g.traverse(finder, nodes, filter=ops.Node)
