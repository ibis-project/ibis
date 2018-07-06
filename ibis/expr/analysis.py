# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import toolz

import ibis.expr.types as ir
import ibis.expr.lineage as lin
import ibis.expr.operations as ops

from ibis.expr.schema import HasSchema
from ibis.expr.window import window

from ibis.common import RelationError, ExpressionError, IbisTypeError


# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_for(expr, substitutions):
    mapping = {repr(k.op()): v for k, v in substitutions}
    substitutor = Substitutor()
    return substitutor.substitute(expr, mapping)


def _expr_key(expr):
    try:
        name = expr.get_name()
    except (AttributeError, ExpressionError):
        name = None

    try:
        op = expr.op()
    except AttributeError:
        return expr, name
    else:
        return repr(op), name


class Substitutor(object):

    def __init__(self):
        cache = toolz.memoize(key=lambda args, kwargs: _expr_key(args[0]))
        self.substitute = cache(self._substitute)

    def _substitute(self, expr, mapping):
        """Substitute expressions with other expressions.

        Parameters
        ----------
        expr : ibis.expr.types.Expr
        mapping : Dict, OrderedDict

        Returns
        -------
        new_expr : ibis.expr.types.Expr
        """
        node = expr.op()
        key = repr(node)
        if key in mapping:
            return mapping[key]
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


class ScalarAggregate(object):

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
        table, = tables
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
    rewriter = ExprSimplifier(expr, lift_memo=lift_memo,
                              block_projection=not past_projection)
    return rewriter.get_result()


class ExprSimplifier(object):

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
                arg, block=self.block_projection)
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
        unchanged = [True]

        def _lift(x):
            if isinstance(x, ir.Expr):
                lifted_arg = self.lift(x, block=block)
                if lifted_arg is not x:
                    unchanged[0] = False
            else:
                # a string or some other thing
                lifted_arg = x
            return lifted_arg

        if arg is None:
            return arg, True

        if isinstance(arg, (tuple, list)):
            result = [_lift(x) for x in arg]
        else:
            result = _lift(arg)

        return result, unchanged[0]

    def lift(self, expr, block=None):
        # This use of id() is OK since only for memoization
        key = id(expr.op()), block

        if key in self.lift_memo:
            return self.lift_memo[key]

        op = expr.op()

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

            for val in root.selections:
                if (isinstance(val.op(), ops.PhysicalTable) and
                        node.name in val.schema()):
                    can_lift = True
                    lifted_root = self.lift(val)
                elif (isinstance(val.op(), ops.TableColumn) and
                      val.op().name == val.get_name() and
                      node.name == val.get_name()):
                    can_lift = True
                    lifted_root = self.lift(val.op().table)

            # HACK: If we've projected a join, do not lift the children
            # TODO: what about limits and other things?
            # if isinstance(root.table.op(), Join):
            #     can_lift = False

            if can_lift and not block:
                lifted_node = ops.TableColumn(node.name, lifted_root)
                result = expr._factory(lifted_node, name=expr._name)

        return result

    def _lift_Aggregation(self, expr, block=None):
        if block is None:
            block = self.block_projection

        op = expr.op()

        # as exposed in #544, do not lift the table inside (which may be
        # filtered or otherwise altered in some way) if blocking

        if block:
            lifted_table = op.table
        else:
            lifted_table = self.lift(op.table, block=True)

        unch = lifted_table is op.table

        lifted_aggs, unch1 = self._lift_arg(op.metrics, block=True)
        lifted_by, unch2 = self._lift_arg(op.by, block=True)
        lifted_having, unch3 = self._lift_arg(op.having, block=True)

        unchanged = unch and unch1 and unch2 and unch3

        if not unchanged:
            lifted_op = ops.Aggregation(lifted_table, lifted_aggs,
                                        by=lifted_by, having=lifted_having)
            result = ir.TableExpr(lifted_op)
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
            lifted_projection = ops.Selection(lifted_table, lifted_selections,
                                              lifted_predicates,
                                              lifted_sort_keys)
            result = ir.TableExpr(lifted_projection)
        else:
            result = expr

        return result

    def _lift_Join(self, expr, block=None):
        op = expr.op()

        left_lifted = self.lift(op.left, block=block)
        right_lifted = self.lift(op.right, block=block)

        unchanged = (left_lifted is op.left and
                     right_lifted is op.right)

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

        helper = ExprSimplifier(expr, lift_memo=self.lift_memo,
                                block_projection=block)
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
            if not has_reduction(predicate) else predicate
            for predicate in predicates
        ]

        if op.table._is_valid(simplified_predicates):
            result = ops.Aggregation(
                op.table, op.metrics, by=op.by, having=op.having,
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys)

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
            if not has_reduction(predicate) else predicate
            for predicate in predicates
        ]

        if op.table._is_valid(simplified_predicates):
            result = ops.Selection(
                op.table, [],
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys)
            return result.to_expr()

    can_pushdown = _can_pushdown(op, predicates)

    if can_pushdown:
        simplified_predicates = [substitute_parents(x) for x in predicates]
        fused_predicates = op.predicates + simplified_predicates
        result = ops.Selection(op.table,
                               selections=op.selections,
                               predicates=fused_predicates,
                               sort_keys=op.sort_keys)
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


class _PushdownValidate(object):

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
            if (isinstance(val.op(), ops.PhysicalTable) and
                    node.name in val.schema()):
                is_valid = True
            elif (isinstance(val.op(), ops.TableColumn) and
                  node.name == val.get_name() and
                  not _is_aliased(val)):
                # Aliased table columns are no good
                col_table = val.op().table.op()

                lifted_node = substitute_parents(expr).op()

                is_valid = (col_table.equals(node.table.op()) or
                            col_table.equals(lifted_node.table.op()))

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
                walked = x._factory(ops.WindowOp(walked_child, window_w),
                                    name=x._name)
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


class Projector(object):

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

        node = self.parent.op()

        if isinstance(node, ops.Selection):
            roots = [node]
        else:
            roots = node.root_tables()

        self.parent_roots = roots

        clean_exprs = []

        for expr in self.resolved_exprs:
            # Perform substitution only if we share common roots
            expr = windowize_function(expr)
            clean_exprs.append(expr)

        self.clean_exprs = clean_exprs

    def get_result(self):
        roots = self.parent_roots
        first_root = roots[0]

        if len(roots) == 1 and isinstance(first_root, ops.Selection):
            fused_op = self._check_fusion(first_root)
            if fused_op is not None:
                return fused_op

        return ops.Selection(self.parent, self.clean_exprs)

    def _check_fusion(self, root):
        roots = root.table._root_tables()
        validator = ExprValidator([root.table])
        fused_exprs = []
        can_fuse = False

        resolved = _maybe_resolve_exprs(root.table, self.input_exprs)
        if not resolved:
            return None

        for val in resolved:
            # XXX
            lifted_val = substitute_parents(val)

            # a * projection
            if (isinstance(val, ir.TableExpr) and
                (self.parent.op().equals(val.op()) or
                 # gross we share the same table root. Better way to
                 # detect?
                 len(roots) == 1 and val._root_tables()[0] is roots[0])):
                can_fuse = True

                have_root = False
                for y in root.selections:
                    # Don't add the * projection twice
                    if y.equals(root.table):
                        fused_exprs.append(root.table)
                        have_root = True
                        continue
                    fused_exprs.append(y)

                # This was a filter, so implicitly a select *
                if not have_root and len(root.selections) == 0:
                    fused_exprs = [root.table] + fused_exprs
            elif validator.validate(lifted_val):
                can_fuse = True
                fused_exprs.append(lifted_val)
            elif not validator.validate(val):
                can_fuse = False
                break
            else:
                fused_exprs.append(val)

        if can_fuse:
            return ops.Selection(root.table, fused_exprs,
                                 predicates=root.predicates,
                                 sort_keys=root.sort_keys)
        else:
            return None


def _maybe_resolve_exprs(table, exprs):
    try:
        return table._resolve(exprs)
    except (AttributeError, IbisTypeError):
        return None


class ExprValidator(object):

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
        return any(self._among_roots(root)
                   for root in expr_roots)

    def shares_one_root(self, expr):
        expr_roots = expr._root_tables()
        total = sum(self.roots_shared(root)
                    for root in expr_roots)
        return total == 1

    def shares_multiple_roots(self, expr):
        expr_roots = expr._root_tables()
        total = sum(self.roots_shared(expr_roots)
                    for root in expr_roots)
        return total > 1

    def validate_all(self, exprs):
        for expr in exprs:
            self.assert_valid(expr)

    def assert_valid(self, expr):
        if not self.validate(expr):
            msg = self._error_message(expr)
            raise RelationError(msg)

    def _error_message(self, expr):
        return ('The expression %s does not fully originate from '
                'dependencies of the table expression.' % repr(expr))


def fully_originate_from(exprs, parents):
    def finder(expr):
        op = expr.op()

        if isinstance(expr, ir.TableExpr):
            return lin.proceed, expr
        elif op.blocks():
            return lin.halt, None
        else:
            return lin.proceed, None

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

        if (isinstance(expr, ir.BooleanColumn) and
                isinstance(op, ops.TableColumn)):
            return True

        is_valid = True

        if isinstance(op, ops.Contains):
            value_valid = super(FilterValidator, self).validate(op.value)
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
        a : double
        b : string
    Selection[table]
      table:
        Table: ref_0
      selections:
        Table: ref_0
        c = Add[double*]
          left:
            a = Column[double*] 'a' from table
              ref_0
          right:
            Literal[double]
              42.0
    >>> find_source_table(expr)
    UnboundTable[table]
      name: t
      schema:
        a : double
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
    options = list(toolz.unique(first_tables, key=id))

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
        if isinstance(op, (ops.Reduction, ops.AnalyticOp)):
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
