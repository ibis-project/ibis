from __future__ import annotations

import collections
import functools
import operator
from collections import Counter
from typing import Sequence

import toolz

import ibis.expr.lineage as lin
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.common.exceptions import (
    ExpressionError,
    IbisTypeError,
    IntegrityError,
)
from ibis.expr.window import window

# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_for(expr, substitutions):
    """Substitute subexpressions in `expr` with expression to expression
    mapping `substitutions`.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        An Ibis expression
    substitutions : List[Tuple[ibis.expr.types.Expr, ibis.expr.types.Expr]]
        A mapping from expression to expression. If any subexpression of `expr`
        is equal to any of the keys in `substitutions`, the value for that key
        will replace the corresponding expression in `expr`.

    Returns
    -------
    ibis.expr.types.Expr
        An Ibis expression
    """

    mapping = {k.op(): v for k, v in substitutions}

    def fn(node):
        try:
            return mapping[node]
        except KeyError:
            if node.blocks():
                return node.to_expr()
            return lin.proceed

    return substitute(fn, expr)


class ScalarAggregate:
    def __init__(self, expr):
        self.expr = expr
        self.tables = []

    def get_result(self):
        expr = self.expr
        subbed_expr = self._visit(expr)

        table = self.tables[0]
        for other in self.tables[1:]:
            table = table.cross_join(other)

        return table.projection([subbed_expr])

    def _visit(self, expr):
        if is_scalar_reduction(expr) and not has_multiple_bases(expr):
            # An aggregation unit
            if not expr.has_name():
                expr = expr.name('tmp')
            agg_expr = reduction_to_aggregation(expr)
            self.tables.append(agg_expr)
            return agg_expr[expr.get_name()]
        elif not isinstance(expr, ir.Expr):
            return expr

        node = expr.op()
        new_expr = node.__class__(*map(self._visit, node.args)).to_expr()
        if expr.has_name():
            new_expr = new_expr.name(name=expr.get_name())

        return new_expr


def has_multiple_bases(expr):
    return toolz.count(find_immediate_parent_tables(expr)) > 1


def reduction_to_aggregation(expr):
    tables = list(find_immediate_parent_tables(expr))

    if len(tables) == 1:
        (table,) = tables
        agg = table.aggregate([expr])
    else:
        agg = ScalarAggregate(expr).get_result()

    return agg


def find_immediate_parent_tables(expr):
    """Find every first occurrence of a :class:`ibis.expr.types.Table`
    object in `expr`.

    Parameters
    ----------
    expr : ir.Expr

    Yields
    ------
    e : ir.Expr

    Notes
    -----
    This function does not traverse into Table objects. This means that the
    underlying PhysicalTable of a Selection will not be yielded, for example.

    Examples
    --------
    >>> import ibis, toolz
    >>> t = ibis.table([('a', 'int64')], name='t')
    >>> expr = t.mutate(foo=t.a + 1)
    >>> result = list(find_immediate_parent_tables(expr))
    >>> len(result)
    1
    >>> result[0]
    r0 := UnboundTable[t]
      a int64
    Selection[r0]
      selections:
        r0
        foo: r0.a + 1
    """

    def finder(expr):
        if isinstance(expr, ir.Table):
            return lin.halt, expr
        else:
            return lin.proceed, None

    return toolz.unique(lin.traverse(finder, expr))


def substitute(fn, expr):
    """Substitute expressions with other expressions."""
    node = expr.op()

    result = fn(node)
    if result is lin.halt:
        return expr
    elif result is not lin.proceed:
        assert isinstance(result, ir.Expr), type(result)
        return result

    new_args = []
    for arg in node.args:
        if isinstance(arg, tuple):
            arg = tuple(
                substitute(fn, expr) if isinstance(arg, ir.Expr) else expr
                for expr in arg
            )
        elif isinstance(arg, ir.Expr):
            arg = substitute(fn, arg)
        new_args.append(arg)

    try:
        new_node = node.__class__(*new_args)
    except IbisTypeError:
        return expr
    else:
        # unfortunately we can't use `.to_expr()` here because it's not backend
        # aware, and some backends have their own ir.Table subclasses, like
        # impala. There's probably a design flaw in the modeling of
        # backend-specific expressions.
        return type(expr)(new_node)


def substitute_parents(expr):
    """
    Rewrite the input expression by replacing any table expressions part of a
    "commutative table operation unit" (for lack of scientific term, a set of
    operations that can be written down in any order and still yield the same
    semantic result)
    """

    def fn(node):
        if isinstance(node, ops.Selection):
            # stop substituting child nodes
            return lin.halt
        elif isinstance(node, ops.TableColumn):
            # For table column references, in the event that we're on top of a
            # projection, we need to check whether the ref comes from the base
            # table schema or is a derived field. If we've projected out of
            # something other than a physical table, then lifting should not
            # occur
            table = node.table.op()

            if isinstance(table, ops.Selection):
                for val in table.selections:
                    if (
                        isinstance(val.op(), ops.PhysicalTable)
                        and node.name in val.schema()
                    ):
                        return ops.TableColumn(val, node.name).to_expr()

        # keep looking for nodes to substitute
        return lin.proceed

    return substitute(fn, expr)


def get_mutation_exprs(
    exprs: list[ir.Expr], table: ir.Table
) -> list[ir.Expr | None]:
    """Given the list of exprs and the underlying table of a mutation op,
    return the exprs to use to instantiate the mutation."""
    # The below logic computes the mutation node exprs by splitting the
    # assignment exprs into two disjoint sets:
    # 1) overwriting_cols_to_expr, which maps a column name to its expr
    # if the expr contains a column that overwrites an existing table column.
    # All keys in this dict are columns in the original table that are being
    # overwritten by an assignment expr. All values in this dict are either:
    #     (a) The expr of the overwriting column. Note that in the case of
    #     DestructColumn, this will specifically only happen for the first
    #     overwriting column within that expr.
    #     (b) None. This is the case for the second (and beyond) overwriting
    #     column(s) inside the DestructColumn and is used as a flag to prevent
    #     the same DestructColumn expr from being duplicated in the output.
    # 2) non_overwriting_exprs, which is a list of all exprs that do not do
    # any overwriting. That is, if an expr is in this list, then its column
    # name does not exist in the original table.
    # Given these two data structures, we can compute the mutation node exprs
    # based on whether any columns are being overwritten.
    # TODO issue #2649
    # Due to a known limitation with how we treat DestructColumn
    # in assignments, the ordering of op.selections may not exactly
    # correspond with the column ordering we want (i.e. all new columns
    # should appear at the end, but currently they are materialized
    # directly after those overwritten columns).
    overwriting_cols_to_expr: dict[str, ir.Expr | None] = {}
    non_overwriting_exprs: list[ir.Expr] = []
    table_schema = table.schema()
    for expr in exprs:
        is_first_overwrite = True
        expr_contains_overwrite = False
        if isinstance(expr, ir.DestructColumn):
            if expr.has_name():
                raise ExpressionError(
                    f"Cannot name a destruct column: {expr.get_name()}"
                )
            for name in expr.type().names:
                if name in table_schema:
                    # The below is necessary to ensure that:
                    # A) all overwritten cols inside the DestructColumn are
                    # accounted for, while
                    # B) we don't repeat the same DestructColumn expr more
                    # than once inside the final mutation node exprs.
                    # This is both okay and necessary because DestructColumn
                    # columns are all packaged together, so the expr should
                    # appear exactly once in the mutation node exprs.
                    if is_first_overwrite:
                        overwriting_cols_to_expr[name] = expr
                        is_first_overwrite = False
                    else:
                        overwriting_cols_to_expr[name] = None
                    expr_contains_overwrite = True
        elif isinstance(expr, ir.Value) and expr.get_name() in table_schema:
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


def apply_filter(expr, predicates):
    # This will attempt predicate pushdown in the cases where we can do it
    # easily and safely, to make both cleaner SQL and fewer referential errors
    # for users

    op = expr.op()

    if isinstance(op, ops.Selection):
        return _filter_selection(expr, predicates)
    elif isinstance(op, ops.Aggregation):
        # Potential fusion opportunity
        # GH1344: We can't sub in things with correlated subqueries
        simplified_predicates = tuple(
            # Originally this line tried substituting op.table in for expr, but
            # that is too aggressive in the presence of filters that occur
            # after aggregations.
            #
            # See https://github.com/ibis-project/ibis/pull/3341 for details
            sub_for(predicate, [(op.table, expr)])
            if not is_reduction(predicate)
            else predicate
            for predicate in predicates
        )

        if shares_all_roots(simplified_predicates, op.table):
            result = ops.Aggregation(
                op.table,
                op.metrics,
                by=op.by,
                having=op.having,
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys,
            )

            return ir.Table(result)

    if not predicates:
        return expr
    return ops.Selection(expr, [], predicates).to_expr()


def _filter_selection(expr, predicates):
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

    op = expr.op()
    # Potential fusion opportunity. The predicates may need to be
    # rewritten in terms of the child table. This prevents the broken
    # ref issue (described in more detail in #59)
    try:
        simplified_predicates = tuple(
            sub_for(predicate, [(expr, op.table)])
            if not is_reduction(predicate)
            else predicate
            for predicate in predicates
        )
    except IntegrityError:
        pass
    else:
        if shares_all_roots(simplified_predicates, op.table) and not any(
            # we can't push down filters on unnest because unnest changes the
            # shape and potential values of the data: unnest can potentially
            # produce NULLs
            #
            # the getattr shenanigans is to handle Alias
            isinstance(
                child_op.arg.op()
                if isinstance(child_op := sel.op(), ops.Alias)
                else child_op,
                ops.Unnest,
            )
            for sel in op.selections
        ):
            result = ops.Selection(
                op.table,
                selections=op.selections,
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys,
            )
            return result.to_expr()

    can_pushdown = _can_pushdown(op, predicates)

    if can_pushdown:
        simplified_predicates = tuple(
            substitute_parents(x) for x in predicates
        )
        fused_predicates = op.predicates + simplified_predicates
        result = ops.Selection(
            op.table,
            selections=op.selections,
            predicates=fused_predicates,
            sort_keys=op.sort_keys,
        )
    else:
        result = ops.Selection(expr, selections=[], predicates=predicates)

    return result.to_expr()


def _can_pushdown(op, predicates):
    # Per issues discussed in #173
    #
    # The only case in which pushdown is possible is that all table columns
    # referenced must meet all of the following (not that onerous in practice)
    # criteria
    #
    # 1) Is a table column, not any other kind of expression
    # 2) Is unaliased. So, if you project t3.foo AS bar, then filter on bar,
    #    this cannot be pushed down (until we implement alias rewriting if
    #    necessary)
    # 3) Appears in the selections in the projection (either is part of one of
    #    the entire tables or a single column selection)

    for pred in predicates:
        validator = _PushdownValidate(op, pred)
        predicate_is_valid = validator.get_result()
        if not predicate_is_valid:
            return False
    return True


class _PushdownValidate:
    def __init__(self, parent, predicate):
        self.parent = parent
        self.pred = predicate

    def get_result(self):
        def validate(expr):
            op = expr.op()
            if isinstance(op, ops.TableColumn):
                return lin.proceed, self._validate_projection(expr)
            return lin.proceed, None

        return all(lin.traverse(validate, self.pred, type=ir.Value))

    def _validate_projection(self, expr):
        is_valid = False
        node = expr.op()

        for val in self.parent.selections:
            if (
                isinstance(val.op(), ops.PhysicalTable)
                and node.name in val.schema()
            ):
                is_valid = True
            elif (
                isinstance(val.op(), ops.TableColumn)
                and node.name == val.get_name()
            ):
                # Aliased table columns are no good
                col_table = val.op().table.op()
                is_valid = col_table.equals(node.table.op())

        return is_valid


def windowize_function(expr, w=None):
    def _windowize(x, w):
        if not isinstance(x.op(), ops.Window):
            walked = _walk(x, w)
        else:
            window_arg, window_w = x.op().args
            walked_child = _walk(window_arg, w)

            if walked_child is not window_arg:
                op = ops.Window(walked_child, window_w)
                walked = op.to_expr().name(x.get_name())
            else:
                walked = x

        op = walked.op()
        if isinstance(op, (ops.Analytic, ops.Reduction)):
            if w is None:
                w = window()
            return walked.over(w)
        elif isinstance(op, ops.Window):
            if w is not None:
                return walked.over(w.combine(op.window))
            else:
                return walked
        else:
            return walked

    def _walk(x, w):
        op = x.op()

        unchanged = True
        windowed_args = []
        for arg in op.args:
            if not isinstance(arg, ir.Value):
                windowed_args.append(arg)
                continue

            new_arg = _windowize(arg, w)
            unchanged = unchanged and arg is new_arg
            windowed_args.append(new_arg)

        if not unchanged:
            new_op = type(op)(*windowed_args)
            expr = new_op.to_expr()
            if x.has_name():
                expr = expr.name(x.get_name())
            return expr
        else:
            return x

    return _windowize(expr, w)


class Projector:

    """
    Analysis and validation of projection operation, taking advantage of
    "projection fusion" opportunities where they exist, i.e. combining
    compatible projections together rather than nesting them. Translation /
    evaluation later will not attempt to do any further fusion /
    simplification.
    """

    def __init__(self, parent, proj_exprs):
        self.parent = parent
        self.input_exprs = proj_exprs
        self.resolved_exprs = [parent._ensure_expr(e) for e in proj_exprs]
        node = parent.op()
        self.parent_roots = (
            [node] if isinstance(node, ops.Selection) else node.root_tables()
        )
        self.clean_exprs = list(map(windowize_function, self.resolved_exprs))

    def get_result(self):
        roots = self.parent_roots
        first_root = roots[0]

        if len(roots) == 1 and isinstance(first_root, ops.Selection):
            fused_op = self.try_fusion(first_root)
            if fused_op is not None:
                return fused_op

        return ops.Selection(self.parent, self.clean_exprs)

    def try_fusion(self, root):
        assert self.parent.op() == root

        root_table = root.table
        roots = root_table.op().root_tables()
        fused_exprs = []
        clean_exprs = self.clean_exprs

        if not isinstance(root_table.op(), ops.Join):
            try:
                resolved = [
                    root_table._ensure_expr(expr)
                    for expr in util.promote_list(self.input_exprs)
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
                and val.op().root_tables()[0] is roots[0]
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
            elif shares_all_roots(val, root_table):
                fused_exprs.append(val)
            else:
                return None

        return ops.Selection(
            root_table,
            fused_exprs,
            predicates=root.predicates,
            sort_keys=root.sort_keys,
        )


def find_first_base_table(expr):
    def predicate(expr):
        op = expr.op()
        if isinstance(op, ops.TableNode):
            return lin.halt, expr
        else:
            return lin.proceed, None

    try:
        return next(lin.traverse(predicate, expr))
    except StopIteration:
        return None


def _find_root_table(expr):
    op = expr.op()

    if isinstance(op, ops.Selection):
        # remove predicates and sort_keys, so that child tables are considered
        # equivalent even if their predicates and sort_keys are not
        return lin.proceed, op._projection
    elif op.blocks():
        return lin.halt, op
    else:
        return lin.proceed, None


def shares_all_roots(exprs, parents):
    # unique table dependencies of exprs and parents
    exprs_deps = set(lin.traverse(_find_root_table, exprs))
    parents_deps = set(lin.traverse(_find_root_table, parents))
    return exprs_deps <= parents_deps


def shares_some_roots(exprs, parents):
    # unique table dependencies of exprs and parents
    exprs_deps = set(lin.traverse(_find_root_table, exprs))
    parents_deps = set(lin.traverse(_find_root_table, parents))
    return bool(exprs_deps & parents_deps)


@util.deprecated(version="4.0", instead="")
def find_source_table(expr):  # pragma: no cover
    """Find the first table expression observed for each argument that the
    expression depends on

    Parameters
    ----------
    expr : ir.Expr

    Returns
    -------
    table_expr : ir.Table

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('a', 'double'), ('b', 'string')], name='t')
    >>> expr = t.mutate(c=t.a + 42.0)
    >>> expr
    r0 := UnboundTable[t]
      a float64
      b string
    Selection[r0]
      selections:
        r0
        c: r0.a + 42.0
    >>> find_source_table(expr)
    UnboundTable[t]
      a float64
      b string
    >>> left = ibis.table([('a', 'int64'), ('b', 'string')])
    >>> right = ibis.table([('c', 'int64'), ('d', 'string')])
    >>> result = left.inner_join(right, left.a == right.c)
    >>> find_source_table(result)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: More than one base table not implemented
    """

    def finder(expr):
        if isinstance(expr, ir.Table):
            return lin.halt, expr
        else:
            return lin.proceed, None

    first_tables = lin.traverse(finder, expr.op().flat_args())
    options = list(toolz.unique(first_tables, key=operator.methodcaller('op')))

    if len(options) > 1:
        raise NotImplementedError('More than one base table not implemented')

    return options[0]


def flatten_predicate(expr):
    """Yield the expressions corresponding to the `And` nodes of a predicate.

    Parameters
    ----------
    expr : ir.BooleanColumn

    Returns
    -------
    exprs : List[ir.BooleanColumn]

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('a', 'int64'), ('b', 'string')], name='t')
    >>> filt = (t.a == 1) & (t.b == 'foo')
    >>> predicates = flatten_predicate(filt)
    >>> len(predicates)
    2
    >>> predicates[0]
    r0 := UnboundTable[t]
      a int64
      b string
    r0.a == 1
    >>> predicates[1]
    r0 := UnboundTable[t]
      a int64
      b string
    r0.b == 'foo'
    """

    def predicate(expr):
        if isinstance(expr.op(), ops.And):
            return lin.proceed, None
        else:
            return lin.halt, expr

    return list(lin.traverse(predicate, expr, type=ir.BooleanColumn))


def _is_ancestor(parent, child):  # pragma: no cover
    """
    Check whether an operation is an ancestor node of another.

    Parameters
    ----------
    parent : ops.Node
    child : ops.Node

    Returns
    -------
    check output : bool
    """

    def predicate(expr):
        if expr.op() == child:
            return lin.halt, True
        else:
            return lin.proceed, None

    return any(lin.traverse(predicate, parent.to_expr()))


def is_analytic(expr):
    def predicate(expr):
        if isinstance(expr.op(), (ops.Reduction, ops.Analytic)):
            return lin.halt, True
        else:
            return lin.proceed, None

    return any(lin.traverse(predicate, expr))


def is_reduction(expr):
    """
    Check whether an expression contains a reduction or not

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

    Parameters
    ----------
    expr : ir.Expr

    Returns
    -------
    check output : bool
    """

    def predicate(expr):
        if isinstance(expr.op(), ops.Reduction):
            return lin.halt, True
        elif isinstance(expr.op(), ops.TableNode):
            # don't go below any table nodes
            return lin.halt, None
        else:
            return lin.proceed, None

    return any(lin.traverse(predicate, expr))


def is_scalar_reduction(expr):
    return isinstance(expr, ir.Scalar) and is_reduction(expr)


_ANY_OP_MAPPING = {
    ops.Any: ops.UnresolvedExistsSubquery,
    ops.NotAny: ops.UnresolvedNotExistsSubquery,
}


def find_predicates(expr, flatten=True):
    def predicate(expr):
        if isinstance(expr, ir.BooleanColumn):
            if flatten and isinstance(expr.op(), ops.And):
                return lin.proceed, None
            else:
                return lin.halt, expr
        return lin.proceed, None

    return list(lin.traverse(predicate, expr))


def find_subqueries(expr: ir.Expr) -> Counter:
    def predicate(
        counts: Counter, expr: ir.Expr
    ) -> tuple[Sequence[ir.Table] | bool, None]:
        op = expr.op()

        if isinstance(op, ops.Join):
            return [op.left, op.right], None
        elif isinstance(op, ops.PhysicalTable):
            return lin.halt, None
        elif isinstance(op, ops.SelfReference):
            return lin.proceed, None
        elif isinstance(op, (ops.Selection, ops.Aggregation)):
            counts[op] += 1
            return [op.table], None
        elif isinstance(op, ops.TableNode):
            counts[op] += 1
            return lin.proceed, None
        elif isinstance(op, ops.TableColumn):
            return op.table.op() not in counts, None
        else:
            return lin.proceed, None

    counts = Counter()
    iterator = lin.traverse(
        functools.partial(predicate, counts),
        expr,
        # keep duplicates so we can determine where an expression is used
        # more than once
        dedup=False,
    )
    # consume the iterator
    collections.deque(iterator, maxlen=0)
    return counts


def _make_any(
    expr,
    any_op_class: type[ops.Any] | type[ops.NotAny],
):
    tables = list(find_immediate_parent_tables(expr))
    predicates = find_predicates(expr, flatten=True)

    if len(tables) > 1:
        op = _ANY_OP_MAPPING[any_op_class](
            tables=tables,
            predicates=predicates,
        )
    else:
        op = any_op_class(expr)
    return op.to_expr()


@functools.singledispatch
def _rewrite_filter(op, _, **kwargs):
    raise NotImplementedError(type(op))


@_rewrite_filter.register(ops.Reduction)
def _rewrite_filter_reduction(_, expr, name: str | None = None, **kwargs):
    """Turn a reduction inside of a filter into an aggregate."""
    # TODO: what about reductions that reference a join that isn't visible at
    # this level? Means we probably have the wrong design, but will have to
    # revisit when it becomes a problem.
    if name is not None:
        expr = expr.name(name)
    aggregation = reduction_to_aggregation(expr)
    return aggregation.to_array()


@_rewrite_filter.register(ops.Any)
@_rewrite_filter.register(ops.TableColumn)
@_rewrite_filter.register(ops.Literal)
@_rewrite_filter.register(ops.ExistsSubquery)
@_rewrite_filter.register(ops.NotExistsSubquery)
@_rewrite_filter.register(ops.Window)
def _rewrite_filter_subqueries(_, expr, **kwargs):
    """Don't rewrite any of these operations in filters."""
    return expr


@_rewrite_filter.register(ops.Alias)
def _rewrite_filter_alias(op, _, name: str | None = None, **kwargs):
    """Rewrite filters on aliases."""
    arg = op.arg
    return _rewrite_filter(
        arg.op(),
        arg,
        name=name if name is not None else op.name,
        **kwargs,
    )


@_rewrite_filter.register(ops.Value)
def _rewrite_filter_value(op, expr, **kwargs):
    """Recursively apply filter rewriting on operations."""
    args = op.args
    visited = [
        _rewrite_filter(arg.op(), arg, **kwargs)
        if isinstance(arg, ir.Expr)
        else arg
        for arg in args
    ]
    if all(map(operator.is_, visited, args)):
        return expr
    else:
        return op.__class__(*visited).to_expr()
