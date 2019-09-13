import operator

import toolz

import ibis.expr.lineage as lin
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.common.exceptions import (
    ExpressionError,
    IbisTypeError,
    RelationError,
)
from ibis.expr.schema import HasSchema
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
    substitutor = Substitutor()
    return substitutor.substitute(expr, mapping)


class Substitutor:
    def __init__(self):
        """Initialize the Substitutor class.

        Notes
        -----
        We need a new cache per substitution call, otherwise we leak state
        across calls and end up incorrectly reusing other substitions' cache.
        """
        cache = toolz.memoize(key=lambda args, kwargs: args[0]._key)
        self.substitute = cache(self._substitute)

    def _substitute(self, expr, mapping):
        """Substitute expressions with other expressions.

        Parameters
        ----------
        expr : ibis.expr.types.Expr
        mapping : Mapping[ibis.expr.operations.Node, ibis.expr.types.Expr]

        Returns
        -------
        ibis.expr.types.Expr
        """
        node = expr.op()
        try:
            return mapping[node]
        except KeyError:
            if node.blocks():
                return expr

            new_args = list(node.args)
            unchanged = True
            for i, arg in enumerate(new_args):
                if isinstance(arg, ir.Expr):
                    new_arg = self.substitute(arg, mapping)
                    unchanged = unchanged and new_arg is arg
                    new_args[i] = new_arg
            if unchanged:
                return expr
            try:
                new_node = type(node)(*new_args)
            except IbisTypeError:
                return expr

            try:
                name = expr.get_name()
            except ExpressionError:
                name = None
            return expr._factory(new_node, name=name)


class ScalarAggregate:
    def __init__(self, expr, memo=None, default_name='tmp'):
        self.expr = expr
        self.memo = memo or {}
        self.tables = []
        self.default_name = default_name

    def get_result(self):
        expr = self.expr
        subbed_expr = self._visit(expr)

        try:
            name = subbed_expr.get_name()
            named_expr = subbed_expr
        except ExpressionError:
            name = self.default_name
            named_expr = subbed_expr.name(self.default_name)

        table = self.tables[0]
        for other in self.tables[1:]:
            table = table.cross_join(other)

        return table.projection([named_expr]), name

    def _visit(self, expr):
        if is_scalar_reduction(expr) and not has_multiple_bases(expr):
            # An aggregation unit
            key = self._key(expr)
            if key not in self.memo:
                agg_expr, name = reduction_to_aggregation(expr)
                self.memo[key] = agg_expr, name
                self.tables.append(agg_expr)
            else:
                agg_expr, name = self.memo[key]
            return agg_expr[name]

        elif not isinstance(expr, ir.Expr):
            return expr

        node = expr.op()
        subbed_args = []
        for arg in node.args:
            if isinstance(arg, (tuple, list)):
                subbed_arg = [self._visit(x) for x in arg]
            else:
                subbed_arg = self._visit(arg)
            subbed_args.append(subbed_arg)

        subbed_node = type(node)(*subbed_args)
        if isinstance(expr, ir.ValueExpr):
            result = expr._factory(subbed_node, name=expr._name)
        else:
            result = expr._factory(subbed_node)

        return result

    def _key(self, expr):
        return repr(expr.op())


def has_multiple_bases(expr):
    return toolz.count(find_immediate_parent_tables(expr)) > 1


def reduction_to_aggregation(expr, default_name='tmp'):
    tables = list(find_immediate_parent_tables(expr))

    try:
        name = expr.get_name()
        named_expr = expr
    except ExpressionError:
        name = default_name
        named_expr = expr.name(default_name)

    if len(tables) == 1:
        (table,) = tables
        return table.aggregate([named_expr]), name
    else:
        return ScalarAggregate(expr, None, default_name).get_result()


def find_immediate_parent_tables(expr):
    """Find every first occurrence of a :class:`ibis.expr.types.TableExpr`
    object in `expr`.

    Parameters
    ----------
    expr : ir.Expr

    Yields
    ------
    e : ir.Expr

    Notes
    -----
    This function does not traverse into TableExpr objects. This means that the
    underlying PhysicalTable of a Selection will not be yielded, for example.

    Examples
    --------
    >>> import ibis, toolz
    >>> t = ibis.table([('a', 'int64')], name='t')
    >>> expr = t.mutate(foo=t.a + 1)
    >>> result = list(find_immediate_parent_tables(expr))
    >>> len(result)
    1
    >>> result[0]  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : int64
    Selection[table]
      table:
        Table: ref_0
      selections:
        Table: ref_0
        foo = Add[int64*]
          left:
            a = Column[int64*] 'a' from table
              ref_0
          right:
            Literal[int8]
              1
    """

    def finder(expr):
        if isinstance(expr, ir.TableExpr):
            return lin.halt, expr
        else:
            return lin.proceed, None

    return lin.traverse(finder, expr)


def substitute_parents(expr, lift_memo=None, past_projection=True):
    rewriter = ExprSimplifier(
        expr, lift_memo=lift_memo, block_projection=not past_projection
    )
    return rewriter.get_result()


class ExprSimplifier:

    """
    Rewrite the input expression by replacing any table expressions part of a
    "commutative table operation unit" (for lack of scientific term, a set of
    operations that can be written down in any order and still yield the same
    semantic result)
    """

    def __init__(self, expr, lift_memo=None, block_projection=False):
        self.expr = expr
        self.lift_memo = lift_memo or {}

        self.block_projection = block_projection

    def get_result(self):
        expr = self.expr
        node = expr.op()
        if isinstance(node, ops.Literal):
            return expr

        # For table column references, in the event that we're on top of a
        # projection, we need to check whether the ref comes from the base
        # table schema or is a derived field. If we've projected out of
        # something other than a physical table, then lifting should not occur
        if isinstance(node, ops.TableColumn):
            result = self._lift_TableColumn(expr, block=self.block_projection)
            if result is not expr:
                return result
        # Temporary hacks around issues addressed in #109
        elif isinstance(node, ops.Selection):
            return self._lift_Selection(expr, block=self.block_projection)
        elif isinstance(node, ops.Aggregation):
            return self._lift_Aggregation(expr, block=self.block_projection)

        unchanged = True

        lifted_args = []
        for arg in node.args:
            lifted_arg, unch_arg = self._lift_arg(
                arg, block=self.block_projection
            )
            lifted_args.append(lifted_arg)

            unchanged = unchanged and unch_arg

        # Do not modify unnecessarily
        if unchanged:
            return expr

        lifted_node = type(node)(*lifted_args)
        if isinstance(expr, ir.ValueExpr):
            result = expr._factory(lifted_node, name=expr._name)
        else:
            result = expr._factory(lifted_node)

        return result

    def _lift_arg(self, arg, block=None):
        changed = 0

        def _lift(expr):
            nonlocal changed

            if isinstance(expr, ir.Expr):
                lifted_arg = self.lift(expr, block=block)
                changed += lifted_arg is not expr
            else:
                # a string or some other thing
                lifted_arg = expr
            return lifted_arg

        if arg is None:
            return arg, True

        if util.is_iterable(arg):
            result = list(map(_lift, arg))
        else:
            result = _lift(arg)

        return result, not changed

    def lift(self, expr, block=None):
        op, _ = key = expr.op(), block

        try:
            return self.lift_memo[key]
        except KeyError:
            pass

        if isinstance(op, ops.ValueOp):
            return self._sub(expr, block=block)
        elif isinstance(op, ops.Selection):
            result = self._lift_Selection(expr, block=block)
        elif isinstance(op, ops.Join):
            result = self._lift_Join(expr, block=block)
        elif isinstance(op, (ops.TableNode, HasSchema)):
            return expr
        else:
            return self._sub(expr, block=block)

        # If we get here, time to record the modified expression in our memo to
        # avoid excessive graph-walking
        self.lift_memo[key] = result
        return result

    def _lift_TableColumn(self, expr, block=None):
        node = expr.op()
        root = node.table.op()
        result = expr

        if isinstance(root, ops.Selection):
            can_lift = False
            all_simple_columns = all(
                isinstance(sel.op(), ops.TableColumn)
                and sel.op().name == sel.get_name()
                for sel in root.selections
                if isinstance(sel, ir.ValueExpr)
                if sel.has_name()
            )

            for val in root.selections:
                value_op = val.op()
                if (
                    isinstance(value_op, ops.PhysicalTable)
                    and node.name in val.schema()
                ):
                    can_lift = True
                    lifted_root = self.lift(val)
                elif (
                    all_simple_columns
                    and isinstance(val, ir.ValueExpr)
                    and val.has_name()
                    and node.name == val.get_name()
                ):
                    can_lift = True
                    lifted_root = self.lift(value_op.table)

            if can_lift and not block:
                lifted_node = ops.TableColumn(node.name, lifted_root)
                result = expr._factory(lifted_node, name=expr._name)

        return result

    def _lift_Aggregation(self, expr, block=None):
        if block is None:
            block = self.block_projection

        op = expr.op()
        table = op.table

        # as exposed in #544, do not lift the table inside (which may be
        # filtered or otherwise altered in some way) if blocking

        if block:
            lifted_table = table
        else:
            lifted_table = self.lift(table, block=True)

        unch = lifted_table is op.table

        lifted_aggs, unch1 = self._lift_arg(op.metrics, block=True)
        lifted_by, unch2 = self._lift_arg(op.by, block=True)
        lifted_having, unch3 = self._lift_arg(op.having, block=True)

        unchanged = unch and unch1 and unch2 and unch3

        if not unchanged:
            lifted_op = ops.Aggregation(
                lifted_table, lifted_aggs, by=lifted_by, having=lifted_having
            )
            result = lifted_op.to_expr()
        else:
            result = expr

        return result

    def _lift_Selection(self, expr, block=None):
        if block is None:
            block = self.block_projection

        op = expr.op()

        if block and op.blocks():
            # GH #549: dig no further
            return expr
        else:
            lifted_table, unch = self._lift_arg(op.table, block=True)

        lifted_selections, unch_sel = self._lift_arg(op.selections, block=True)
        unchanged = unch and unch_sel

        lifted_predicates, unch_sel = self._lift_arg(op.predicates, block=True)
        unchanged = unch and unch_sel

        lifted_sort_keys, unch_sel = self._lift_arg(op.sort_keys, block=True)
        unchanged = unch and unch_sel

        if not unchanged:
            lifted_projection = ops.Selection(
                lifted_table,
                lifted_selections,
                lifted_predicates,
                lifted_sort_keys,
            )
            result = lifted_projection.to_expr()
        else:
            result = expr

        return result

    def _lift_Join(self, expr, block=None):
        op = expr.op()

        left_lifted = self.lift(op.left, block=block)
        right_lifted = self.lift(op.right, block=block)

        unchanged = left_lifted is op.left and right_lifted is op.right

        # Fix predicates
        lifted_preds = []
        for x in op.predicates:
            subbed = self._sub(x, block=True)
            if subbed is not x:
                unchanged = False
            lifted_preds.append(subbed)

        if not unchanged:
            lifted_join = type(op)(left_lifted, right_lifted, lifted_preds)
            result = ir.TableExpr(lifted_join)
        else:
            result = expr

        return result

    def _sub(self, expr, block=None):
        # catchall recursive rewriter
        if block is None:
            block = self.block_projection

        helper = ExprSimplifier(
            expr, lift_memo=self.lift_memo, block_projection=block
        )
        return helper.get_result()


def _base_table(table_node):
    # Find the aggregate or projection root. Not proud of this
    if table_node.blocks():
        return table_node
    else:
        return _base_table(table_node.table.op())


def has_reduction(expr):
    """Does `expr` contain a reduction?

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        An ibis expression

    Returns
    -------
    truth_value : bool
        Whether or not there's at least one reduction in `expr`

    Notes
    -----
    The ``isinstance(op, ops.TableNode)`` check in this function implies
    that we only examine every non-table expression that precedes the first
    table expression.
    """

    def fn(expr):
        op = expr.op()
        if isinstance(op, ops.TableNode):  # don't go below any table nodes
            return lin.halt, None
        if isinstance(op, ops.Reduction):
            return lin.halt, True
        return lin.proceed, None

    reduction_status = lin.traverse(fn, expr)
    return any(reduction_status)


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
        simplified_predicates = [
            sub_for(predicate, [(expr, op.table)])
            if not has_reduction(predicate)
            else predicate
            for predicate in predicates
        ]

        if op.table._is_valid(simplified_predicates):
            result = ops.Aggregation(
                op.table,
                op.metrics,
                by=op.by,
                having=op.having,
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys,
            )

            return ir.TableExpr(result)
    elif isinstance(op, ops.Join):
        expr = expr.materialize()

    result = ops.Selection(expr, [], predicates)
    return ir.TableExpr(result)


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
    if not op.blocks():
        # Potential fusion opportunity. The predicates may need to be
        # rewritten in terms of the child table. This prevents the broken
        # ref issue (described in more detail in #59)
        simplified_predicates = [
            sub_for(predicate, [(expr, op.table)])
            if not has_reduction(predicate)
            else predicate
            for predicate in predicates
        ]

        if op.table._is_valid(simplified_predicates):
            result = ops.Selection(
                op.table,
                [],
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys,
            )
            return result.to_expr()

    can_pushdown = _can_pushdown(op, predicates)

    if can_pushdown:
        simplified_predicates = [substitute_parents(x) for x in predicates]
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
        self.validator = ExprValidator([self.parent.table])

    def get_result(self):
        predicate = self.pred
        return not has_reduction(predicate) and all(self._walk(predicate))

    def _walk(self, expr):
        def validate(expr):
            op = expr.op()
            if isinstance(op, ops.TableColumn):
                return lin.proceed, self._validate_column(expr)
            return lin.proceed, None

        return lin.traverse(validate, expr, type=ir.ValueExpr)

    def _validate_column(self, expr):
        if isinstance(self.parent, ops.Selection):
            return self._validate_projection(expr)
        else:
            validator = ExprValidator([self.parent.table])
            return validator.validate(expr)

    def _validate_projection(self, expr):
        is_valid = False
        node = expr.op()

        # Has a different alias, invalid
        if _is_aliased(expr):
            return False

        for val in self.parent.selections:
            if (
                isinstance(val.op(), ops.PhysicalTable)
                and node.name in val.schema()
            ):
                is_valid = True
            elif (
                isinstance(val.op(), ops.TableColumn)
                and node.name == val.get_name()
                and not _is_aliased(val)
            ):
                # Aliased table columns are no good
                col_table = val.op().table.op()

                lifted_node = substitute_parents(expr).op()

                is_valid = col_table.equals(
                    node.table.op()
                ) or col_table.equals(lifted_node.table.op())

        return is_valid


def _is_aliased(col_expr):
    return col_expr.op().name != col_expr.get_name()


def windowize_function(expr, w=None):
    def _windowize(x, w):
        if not isinstance(x.op(), ops.WindowOp):
            walked = _walk(x, w)
        else:
            window_arg, window_w = x.op().args
            walked_child = _walk(window_arg, w)

            if walked_child is not window_arg:
                walked = x._factory(
                    ops.WindowOp(walked_child, window_w), name=x._name
                )
            else:
                walked = x

        op = walked.op()
        if isinstance(op, ops.AnalyticOp) or getattr(op, '_reduction', False):
            if w is None:
                w = window()
            return walked.over(w)
        elif isinstance(op, ops.WindowOp):
            if w is not None:
                return walked.over(w)
            else:
                return walked
        else:
            return walked

    def _walk(x, w):
        op = x.op()

        unchanged = True
        windowed_args = []
        for arg in op.args:
            if not isinstance(arg, ir.ValueExpr):
                windowed_args.append(arg)
                continue

            new_arg = _windowize(arg, w)
            unchanged = unchanged and arg is new_arg
            windowed_args.append(new_arg)

        if not unchanged:
            new_op = type(op)(*windowed_args)
            return x._factory(new_op, name=x._name)
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
        root_table = root.table
        roots = root_table._root_tables()
        validator = ExprValidator([root_table])
        fused_exprs = []
        can_fuse = False
        clean_exprs = self.clean_exprs

        if not isinstance(root_table.op(), ops.Join):
            try:
                resolved = root_table._resolve(self.input_exprs)
            except (AttributeError, IbisTypeError):
                resolved = clean_exprs
        else:
            # joins cannot be used to resolve expressions, but we still may be
            # able to fuse columns from a projection off of a join. In that
            # case, use the projection's input expressions as the columns with
            # which to attempt fusion
            resolved = clean_exprs

        if not resolved:
            return None

        root_selections = root.selections
        parent_op = self.parent.op()
        for val in resolved:
            # XXX
            lifted_val = substitute_parents(val)

            # a * projection
            if isinstance(val, ir.TableExpr) and (
                parent_op.compatible_with(val.op())
                # gross we share the same table root. Better way to
                # detect?
                or len(roots) == 1
                and val._root_tables()[0] is roots[0]
            ):
                can_fuse = True
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
                    fused_exprs = [root_table] + fused_exprs
            elif validator.validate(lifted_val):
                can_fuse = True
                fused_exprs.append(lifted_val)
            elif not validator.validate(val):
                can_fuse = False
                break
            else:
                fused_exprs.append(val)

        if can_fuse:
            return ops.Selection(
                root_table,
                fused_exprs,
                predicates=root.predicates,
                sort_keys=root.sort_keys,
            )
        return None


class ExprValidator:
    def __init__(self, exprs):
        self.parent_exprs = exprs

        self.roots = []
        for expr in self.parent_exprs:
            self.roots.extend(expr._root_tables())

    def has_common_roots(self, expr):
        return self.validate(expr)

    def validate(self, expr):
        op = expr.op()
        if isinstance(op, ops.TableColumn):
            if self._among_roots(op.table.op()):
                return True
        elif isinstance(op, ops.Selection):
            if self._among_roots(op):
                return True

        expr_roots = expr._root_tables()
        for root in expr_roots:
            if not self._among_roots(root):
                return False
        return True

    def _among_roots(self, node):
        return self.roots_shared(node) > 0

    def roots_shared(self, node):
        return sum(root.is_ancestor(node) for root in self.roots)

    def shares_some_roots(self, expr):
        expr_roots = expr._root_tables()
        return any(self._among_roots(root) for root in expr_roots)

    def shares_one_root(self, expr):
        expr_roots = expr._root_tables()
        total = sum(self.roots_shared(root) for root in expr_roots)
        return total == 1

    def shares_multiple_roots(self, expr):
        expr_roots = expr._root_tables()
        total = sum(self.roots_shared(expr_roots) for root in expr_roots)
        return total > 1

    def validate_all(self, exprs):
        for expr in exprs:
            self.assert_valid(expr)

    def assert_valid(self, expr):
        if not self.validate(expr):
            msg = self._error_message(expr)
            raise RelationError(msg)

    def _error_message(self, expr):
        return (
            'The expression %s does not fully originate from '
            'dependencies of the table expression.' % repr(expr)
        )


def fully_originate_from(exprs, parents):
    def finder(expr):
        op = expr.op()

        if isinstance(expr, ir.TableExpr):
            return lin.proceed, expr.op()
        return lin.halt if op.blocks() else lin.proceed, None

    # unique table dependencies of exprs and parents
    exprs_deps = set(lin.traverse(finder, exprs))
    parents_deps = set(lin.traverse(finder, parents))
    return exprs_deps <= parents_deps


class FilterValidator(ExprValidator):

    """
    Filters need not necessarily originate fully from the ancestors of the
    table being filtered. The key cases for this are

    - Scalar reductions involving some other tables
    - Array expressions involving other tables only (mapping to "uncorrelated
      subqueries" in SQL-land)
    - Reductions or array expressions like the above, but containing some
      predicate with a record-specific interdependency ("correlated subqueries"
      in SQL)
    """

    def validate(self, expr):
        op = expr.op()

        if isinstance(expr, ir.BooleanColumn) and isinstance(
            op, ops.TableColumn
        ):
            return True

        is_valid = True

        if isinstance(op, ops.Contains):
            value_valid = super().validate(op.value)
            is_valid = value_valid
        else:
            roots_valid = []
            for arg in op.flat_args():
                if isinstance(arg, ir.ScalarExpr):
                    # arg_valid = True
                    pass
                elif isinstance(arg, ir.TopKExpr):
                    # TopK not subjected to further analysis for now
                    roots_valid.append(True)
                elif isinstance(arg, (ir.ColumnExpr, ir.AnalyticExpr)):
                    roots_valid.append(self.shares_some_roots(arg))
                elif isinstance(arg, ir.Expr):
                    raise NotImplementedError(repr((type(expr), type(arg))))
                else:
                    # arg_valid = True
                    pass

            is_valid = any(roots_valid)

        return is_valid


def find_source_table(expr):
    """Find the first table expression observed for each argument that the
    expression depends on

    Parameters
    ----------
    expr : ir.Expr

    Returns
    -------
    table_expr : ir.TableExpr

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('a', 'double'), ('b', 'string')], name='t')
    >>> expr = t.mutate(c=t.a + 42.0)
    >>> expr  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : float64
        b : string
    Selection[table]
      table:
        Table: ref_0
      selections:
        Table: ref_0
        c = Add[float64*]
          left:
            a = Column[float64*] 'a' from table
              ref_0
          right:
            Literal[float64]
              42.0
    >>> find_source_table(expr)
    UnboundTable[table]
      name: t
      schema:
        a : float64
        b : string
    >>> left = ibis.table([('a', 'int64'), ('b', 'string')])
    >>> right = ibis.table([('c', 'int64'), ('d', 'string')])
    >>> result = left.inner_join(right, left.a == right.c)
    >>> find_source_table(result)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: More than one base table not implemented
    """

    def finder(expr):
        if isinstance(expr, ir.TableExpr):
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
    >>> predicates[0]  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : int64
        b : string
    Equals[boolean*]
      left:
        a = Column[int64*] 'a' from table
          ref_0
      right:
        Literal[int64]
          1
    >>> predicates[1]  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : int64
        b : string
    Equals[boolean*]
      left:
        b = Column[string*] 'b' from table
          ref_0
      right:
        Literal[string]
          foo
    """

    def predicate(expr):
        if isinstance(expr.op(), ops.And):
            return lin.proceed, None
        else:
            return lin.halt, expr

    return list(lin.traverse(predicate, expr, type=ir.BooleanColumn))


def is_analytic(expr, exclude_windows=False):
    def _is_analytic(op):
        if isinstance(op, (ops.Reduction, ops.AnalyticOp, ops.Any, ops.All)):
            return True
        elif isinstance(op, ops.WindowOp) and exclude_windows:
            return False

        for arg in op.args:
            if isinstance(arg, ir.Expr) and _is_analytic(arg.op()):
                return True

        return False

    return _is_analytic(expr.op())


def is_reduction(expr):
    """Check whether an expression is a reduction or not

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

    def has_reduction(op):
        if getattr(op, '_reduction', False):
            return True

        for arg in op.args:
            if isinstance(arg, ir.ScalarExpr) and has_reduction(arg.op()):
                return True

        return False

    return has_reduction(expr.op() if isinstance(expr, ir.Expr) else expr)


def is_scalar_reduction(expr):
    return isinstance(expr, ir.ScalarExpr) and is_reduction(expr)
