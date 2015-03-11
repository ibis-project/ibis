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

from ibis.common import RelationError
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.util as util

#----------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_for(expr, substitutions):
    helper = _Substitutor(expr, substitutions)
    return helper.get_result()


class _Substitutor(object):

    def __init__(self, expr, substitutions, sub_memo=None):
        self.expr = expr

        self.substitutions = substitutions

        self._id_to_expr = {}
        for k, v in substitutions:
            self._id_to_expr[self._key(k)] = v

        self.sub_memo = sub_memo or {}
        self.unchanged = True

    def get_result(self):
        expr = self.expr
        node = expr.op()

        subbed_args = []
        for arg in node.args:
            if isinstance(arg, (tuple, list)):
                subbed_arg = [self._sub_arg(x) for x in arg]
            else:
                subbed_arg = self._sub_arg(arg)
            subbed_args.append(subbed_arg)

        # Do not modify unnecessarily
        if self.unchanged:
            return expr

        subbed_node = type(node)(*subbed_args)
        if isinstance(expr, ir.ValueExpr):
            result = expr._factory(subbed_node, name=expr._name)
        else:
            result = expr._factory(subbed_node)

        return result

    def _sub_arg(self, arg):
        if isinstance(arg, ir.Expr):
            subbed_arg = self.sub(arg)
            if subbed_arg is not arg:
                self.unchanged = False
        else:
            # a string or some other thing
            subbed_arg = arg

        return subbed_arg

    def _key(self, expr):
        return id(expr.op())

    def sub(self, expr):
        key = self._key(expr)

        if key in self.sub_memo:
            return self.sub_memo[key]

        if key in self._id_to_expr:
            return self._id_to_expr[key]

        result = self._sub(expr)

        self.sub_memo[key] = result
        return result

    def _sub(self, expr):
        helper = _Substitutor(expr, self.substitutions,
                              sub_memo=self.sub_memo)
        return helper.get_result()


def substitute_parents(expr, lift_memo=None):
    rewriter = ExprSimplifier(expr, lift_memo=lift_memo)
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
        elif isinstance(node, ops.Projection):
            return self._lift_Projection(expr, block=self.block_projection)
        elif isinstance(node, ops.Aggregation):
            return self._lift_Aggregation(expr, block=self.block_projection)

        unchanged = True

        lifted_args = []
        for arg in node.args:
            lifted_arg, unch_arg = self._lift_arg(arg)
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
        key = id(expr.op()), block
        if key in self.lift_memo:
            return self.lift_memo[key]

        op = expr.op()

        if isinstance(op, (ops.ValueNode, ops.ArrayNode)):
            return self._sub(expr, block=block)
        elif isinstance(op, ops.Filter):
            result = self.lift(op.table, block=block)
        elif isinstance(op, ops.Projection):
            result = self._lift_Projection(expr, block=block)
        elif isinstance(op, ops.Join):
            result = self._lift_Join(expr, block=block)
        elif isinstance(op, (ops.TableNode, ir.HasSchema)):
            return expr
        else:
            raise NotImplementedError

        # If we get here, time to record the modified expression in our memo to
        # avoid excessive graph-walking
        self.lift_memo[key] = result
        return result

    def _lift_TableColumn(self, expr, block=None):
        node = expr.op()

        tnode = node.table.op()
        root = _base_table(tnode)

        result = expr
        if isinstance(root, ops.Projection):
            can_lift = False

            for val in root.selections:
                if (isinstance(val.op(), ops.PhysicalTable) and
                        node.name in val.schema()):

                    can_lift = True
                    lifted_root = self.lift(val)
                elif (isinstance(val.op(), ops.TableColumn)
                      and val.op().name == val.get_name()
                      and node.name == val.get_name()):
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

    def _lift_Projection(self, expr, block=None):
        if block is None:
            block = self.block_projection

        op = expr.op()

        if block:
            lifted_table = op.table
            unch = True
        else:
            lifted_table, unch = self._lift_arg(op.table, block=True)

        lifted_selections, unch_sel = self._lift_arg(op.selections, block=True)
        unchanged = unch and unch_sel
        if not unchanged:
            lifted_projection = ops.Projection(lifted_table, lifted_selections)
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
    if isinstance(table_node, ir.BlockingTableNode):
        return table_node
    else:
        return _base_table(table_node.table.op())


def apply_filter(expr, predicates):
    # This will attempt predicate pushdown in the cases where we can do it
    # easily and safely

    op = expr.op()

    if isinstance(op, ops.Filter):
        # Potential fusion opportunity. The predicates may need to be rewritten
        # in terms of the child table. This prevents the broken ref issue
        # (described in more detail in #59)
        predicates = [sub_for(x, [(expr, op.table)]) for x in predicates]
        return ops.Filter(op.table, op.predicates + predicates)

    elif isinstance(op, (ops.Projection, ops.Aggregation)):
        # if any of the filter predicates have the parent expression among
        # their roots, then pushdown (at least of that predicate) is not
        # possible

        # TODO: is partial pushdown (one or more, but not all of the passed
        # predicates) something we should consider doing? Could be reasonable

        can_pushdown = True
        for pred in predicates:
            roots = pred._root_tables()
            if _in_roots(expr, roots):
                can_pushdown = False

        # It's not unusual for the filter to reference the projection
        # itself. If a predicate can be pushed down, in this case we must
        # rewrite replacing the table refs with the roots internal to the
        # projection we are referencing
        #
        # in pseudocode
        # c = Projection(Join(a, b, jpreds), ppreds)
        # filter_pred = c.field1 == c.field2
        # Filter(c, [filter_pred])
        #
        # Assuming that the fields referenced by the filter predicate originate
        # below the projection, we need to rewrite the predicate referencing
        # the parent tables in the join being projected

        if can_pushdown:
            predicates = [substitute_parents(x) for x in predicates]

            # this will further fuse, if possible
            filtered = op.table.filter(predicates)
            result = op.substitute_table(filtered)
        else:
            result = ops.Filter(expr, predicates)
    else:
        result = ops.Filter(expr, predicates)

    return result

# def _pushdown_substitute(expr):
#     rewriter = _PushdownRewrite(expr)
#     return rewriter.get_result()
# class _PushdownRewrite(object):
#     # Hm, this is quite similar to the ExprSimplifier above
#     def __init__(self, expr):
#         self.expr = expr
#     def get_result(self):
#         return self._rewrite(expr)
#     def _rewrite(self, expr):
#         node = expr.op()
#         unchanged = True
#         new_args = []
#         for arg in node.args:
#             pass


def _in_roots(expr, roots):
    # XXX
    what = expr.op() if isinstance(expr, ir.Expr) else expr
    return id(what) in [id(x) for x in roots]


def _maybe_fuse_projection(expr, clean_exprs):
    node = expr.op()

    if isinstance(node, ops.Projection):
        roots = [node]
    else:
        roots = node.root_tables()

    if len(roots) == 1 and isinstance(roots[0], ops.Projection):
        root = roots[0]

        roots = root.root_tables()
        validator = ExprValidator([ir.TableExpr(root)])
        fused_exprs = []
        can_fuse = False
        for val in clean_exprs:
            # a * projection
            if (isinstance(val, ir.TableExpr) and
                (val is expr or

                     # gross we share the same table root. Better way to detect?
                     len(roots) == 1 and val._root_tables()[0] is roots[0])
                    ):
                can_fuse = True
                fused_exprs.extend(root.selections)
            elif not validator.validate(val):
                can_fuse = False
                break
            else:
                fused_exprs.append(val)

        if can_fuse:
            return ops.Projection(root.table, fused_exprs)

    return ops.Projection(expr, clean_exprs)


class ExprValidator(object):

    def __init__(self, exprs):
        self.parent_exprs = exprs

        self.roots = []
        for expr in self.parent_exprs:

            self.roots.extend(expr._root_tables())

        self.root_ids = set(id(x) for x in self.roots)

    def validate(self, expr):
        return self.has_common_roots(expr)

    def has_common_roots(self, expr):
        op = expr.op()
        if isinstance(op, ops.TableColumn):
            for root in self.roots:
                if root is op.table.op():
                    return True
        elif isinstance(op, ops.Projection):
            for root in self.roots:
                if root is op:
                    return True

        expr_roots = expr._root_tables()
        for root in expr_roots:
            if id(root) not in self.root_ids:
                return False
        return True

    def shares_some_roots(self, expr):
        expr_roots = expr._root_tables()
        return any(id(root) in self.root_ids for root in expr_roots)

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
            value_valid = self.has_common_roots(op.value)
            is_valid = value_valid
        else:
            roots_valid = []
            for arg in op.flat_args():
                if isinstance(arg, ir.ScalarExpr):
                    arg_valid = True
                elif isinstance(arg, ir.ArrayExpr):
                    roots_valid.append(self.shares_some_roots(arg))
                elif isinstance(arg, ir.Expr):
                    raise NotImplementedError
                else:
                    arg_valid = True

                # args_valid.append(arg_valid)

            is_valid = any(roots_valid)

        return is_valid


def find_base_table(expr):
    if isinstance(expr, ir.TableExpr):
        return expr

    for arg in expr.op().args:
        if isinstance(arg, ir.Expr):
            r = find_base_table(arg)
            if isinstance(r, ir.TableExpr):
                return r


def find_source_table(expr):
    # A more complex version of _find_base_table.
    # TODO: Revisit/refactor this all at some point
    node = expr.op()

    # First table expression observed for each argument that the expr
    # depends on
    first_tables = []

    def push_first(arg):
        if isinstance(arg, (tuple, list)):
            [push_first(x) for x in arg]
            return

        if not isinstance(arg, ir.Expr):
            return
        if isinstance(arg, ir.TableExpr):
            first_tables.append(arg)
        else:
            collect(arg.op())

    def collect(node):
        for arg in node.args:
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
            raise Exception('Invalid predicate: {!r}'.format(expr))

    walk(expr)
    return out_exprs


def find_backend(expr):
    from ibis.connection import Connection

    backends = []

    def walk(expr):
        node = expr.op()
        for arg in node.flat_args():
            if isinstance(arg, Connection):
                backends.append(arg)
            elif isinstance(arg, ir.Expr):
                walk(arg)

    walk(expr)
    backends = util.unique_by_key(backends, id)

    if len(backends) > 1:
        raise ValueError('Multiple backends found')

    return backends[0]
