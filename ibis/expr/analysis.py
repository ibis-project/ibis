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

from ibis.common import RelationError, ExpressionError, IbisTypeError
from ibis.expr.datatypes import HasSchema
from ibis.expr.window import window
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.util as util
import toolz

# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_for(expr, substitutions):
    mapping = dict((repr(k.op()), v) for k, v in substitutions)
    return _subs(expr, mapping)


def _expr_key(expr):
    try:
        return repr(expr.op())
    except AttributeError:
        return expr


@toolz.memoize(key=lambda args, kwargs: _expr_key(args[0]))
def _subs(expr, mapping):
    """Substitute expressions with other expressions
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
            new_arg = _subs(arg, mapping)
            unchanged = unchanged and new_arg is arg
            new_args[i] = new_arg
    if unchanged:
        return expr
    try:
        new_node = type(node)(*new_args)
    except IbisTypeError:
        return expr

    return expr._factory(new_node, name=getattr(expr, '_name', None))


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
        except:
            name = self.default_name
            named_expr = subbed_expr.name(self.default_name)

        tables = list(self.memo.values())
        table = self.tables[0]
        for other in self.tables[1:]:
            table = table.cross_join(other)

        return table.projection([named_expr]), name

    def _visit(self, expr):
        if is_scalar_reduce(expr) and not has_multiple_bases(expr):
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
    return len(find_all_tables(expr)) > 1


def reduction_to_aggregation(expr, default_name='tmp'):
    tables = find_all_tables(expr)

    try:
        name = expr.get_name()
        named_expr = expr
    except:
        name = default_name
        named_expr = expr.name(default_name)

    if len(tables) == 1:
        table = list(tables.values())[0]
        return table.aggregate([named_expr]), name
    else:
        return ScalarAggregate(expr, None, default_name).get_result()


def find_all_tables(expr, memo=None):
    if memo is None:
        memo = {}

    node = expr.op()

    if isinstance(expr, ir.TableExpr):
        key = id(node)
        if key not in memo:
            memo[key] = expr
        return memo

    for arg in node.flat_args():
        if isinstance(arg, ir.Expr):
            find_all_tables(arg, memo)

    return memo


def is_scalar_reduce(x):
    return isinstance(x, ir.ScalarExpr) and ops.is_reduction(x)


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
        if isinstance(node, ir.Literal):
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

        if isinstance(op, ops.ValueNode):
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

        tnode = node.table.op()
        root = _base_table(tnode)

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

                # XXX
                # can_lift = False

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

        lifted_aggs, unch1 = self._lift_arg(op.agg_exprs, block=True)
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


def apply_filter(expr, predicates):
    # This will attempt predicate pushdown in the cases where we can do it
    # easily and safely, to make both cleaner SQL and fewer referential errors
    # for users

    op = expr.op()

    if isinstance(op, ops.Selection):
        return _filter_selection(expr, predicates)
    elif isinstance(op, ops.Aggregation):
        # Potential fusion opportunity
        simplified_predicates = [
            sub_for(predicate, [(expr, op.table)]) for predicate in predicates
        ]

        if op.table._is_valid(simplified_predicates):
            result = ops.Aggregation(
                op.table, op.agg_exprs, by=op.by, having=op.having,
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
        simplified_predicates = [sub_for(x, [(expr, op.table)])
                                 for x in predicates]

        if op.table._is_valid(simplified_predicates):
            result = ops.Selection(
                op.table, [],
                predicates=op.predicates + simplified_predicates,
                sort_keys=op.sort_keys)
            return ir.TableExpr(result)

    can_pushdown = _can_pushdown(op, predicates)

    if can_pushdown:
        simplified_predicates = [substitute_parents(x) for x in predicates]
        fused_predicates = op.predicates + simplified_predicates
        result = ops.Selection(op.table,
                               proj_exprs=op.selections,
                               predicates=fused_predicates,
                               sort_keys=op.sort_keys)
    else:
        result = ops.Selection(expr, proj_exprs=[],
                               predicates=predicates)

    return ir.TableExpr(result)


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

    can_pushdown = True
    for pred in predicates:
        validator = _PushdownValidate(op, pred)
        predicate_is_valid = validator.get_result()
        can_pushdown = can_pushdown and predicate_is_valid
    return can_pushdown


class _PushdownValidate(object):

    def __init__(self, parent, predicate):
        self.parent = parent
        self.pred = predicate

        self.validator = ExprValidator([self.parent.table])

        self.valid = True

    def get_result(self):
        self._walk(self.pred)
        return self.valid

    def _walk(self, expr):
        node = expr.op()
        if isinstance(node, ops.TableColumn):
            is_valid = self._validate_column(expr)
            self.valid = self.valid and is_valid

        for arg in node.flat_args():
            if isinstance(arg, ir.ValueExpr):
                self._walk(arg)
            # Skip other types of exprs

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

                is_valid = (col_table.is_ancestor(node.table) or
                            col_table.is_ancestor(lifted_node.table))

                # is_valid = True

        return is_valid


def _is_aliased(col_expr):
    return col_expr.op().name != col_expr.get_name()


def windowize_function(expr, w=None):
    def _check_window(x):
        # Hmm
        arg, window = x.op().args
        if isinstance(arg.op(), ops.RowNumber):
            if len(window._order_by) == 0:
                raise ExpressionError('RowNumber requires explicit '
                                      'window sort')

        return x

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
        if (isinstance(op, ops.AnalyticOp) or
                getattr(op, '_reduction', False)):
            if w is None:
                w = window()
            return _check_window(walked.over(w))
        elif isinstance(op, ops.WindowOp):
            if w is not None:
                return _check_window(walked.over(w))
            else:
                return _check_window(walked)
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

        if len(roots) == 1 and isinstance(roots[0], ops.Selection):
            fused_op = self._check_fusion(roots[0])
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
                (self.parent.op().is_ancestor(val) or
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
    except:
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
        count = 0
        for root in self.roots:
            if root.is_ancestor(node):
                count += 1
        return count

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


class CommonSubexpr(object):

    def __init__(self, exprs):
        self.parent_exprs = exprs

    def validate(self, expr):
        if isinstance(expr, ir.TableExpr):
            if not self._check(expr):
                return False

        op = expr.op()

        for arg in op.flat_args():
            if not isinstance(arg, ir.Expr):
                continue
            elif not isinstance(arg, ir.TableExpr):
                if not self.validate(arg):
                    return False
            else:
                # Table expression. Must be found in a parent table expr a
                # blocking root of one of the parent tables
                if not self._check(arg):
                    return False

        return True

    def _check(self, expr):
        # Table dependency matches one of the parent exprs
        is_valid = False
        for parent in self.parent_exprs:
            is_valid = is_valid or self._check_table(parent, expr)
        return is_valid

    def _check_table(self, parent, needle):
        def _matches(expr):
            op = expr.op()

            if expr.equals(needle):
                return True

            if op.blocks():
                return False

            for arg in op.flat_args():
                if not isinstance(arg, ir.Expr):
                    continue
                if _matches(arg):
                    return True

            return True

        return _matches(parent)

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

        is_valid = True

        if isinstance(op, ops.Contains):
            value_valid = ExprValidator.validate(self, op.value)
            is_valid = value_valid
        else:
            roots_valid = []
            for arg in op.flat_args():
                if isinstance(arg, ir.ScalarExpr):
                    # arg_valid = True
                    pass
                elif isinstance(arg, ops.TopKExpr):
                    # TopK not subjected to further analysis for now
                    roots_valid.append(True)
                elif isinstance(arg, (ir.ArrayExpr, ir.AnalyticExpr)):
                    roots_valid.append(self.shares_some_roots(arg))
                elif isinstance(arg, ir.Expr):
                    raise NotImplementedError
                else:
                    # arg_valid = True
                    pass

            is_valid = any(roots_valid)

        return is_valid


def find_source_table(expr):
    # A more complex version of _find_base_table.
    # TODO: Revisit/refactor this all at some point
    node = expr.op()

    # First table expression observed for each argument that the expr
    # depends on
    first_tables = []

    def push_first(arg):
        if not isinstance(arg, ir.Expr):
            return
        if isinstance(arg, ir.TableExpr):
            first_tables.append(arg)
        else:
            collect(arg.op())

    def collect(node):
        for arg in node.flat_args():
            push_first(arg)

    collect(node)
    options = util.unique_by_key(first_tables, id)

    if len(options) > 1:
        raise NotImplementedError

    return options[0]


def unwrap_ands(expr):
    out_exprs = []

    def walk(expr):
        op = expr.op()
        if isinstance(op, ops.Comparison):
            out_exprs.append(expr)
        elif isinstance(op, ops.And):
            walk(op.left)
            walk(op.right)
        else:
            raise Exception('Invalid predicate: {0!s}'
                            .format(expr._repr()))

    walk(expr)
    return out_exprs
