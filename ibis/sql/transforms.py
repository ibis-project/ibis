# Copyright 2015 Cloudera Inc.
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


import ibis.util as util
import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.analysis as L
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.signature import Argument as Arg


class ExistsExpr(ir.AnalyticExpr):

    def type(self):
        return 'exists'


class ExistsSubquery(ops.Node):
    """Helper class"""
    foreign_table = Arg(rlz.noop)
    predicates = Arg(rlz.noop)

    def output_type(self):
        return ExistsExpr


class NotExistsSubquery(ops.Node):
    foreign_table = Arg(rlz.noop)
    predicates = Arg(rlz.noop)

    def output_type(self):
        return ExistsExpr


class AnyToExistsTransform(object):

    """
    Some code duplication with the correlated ref check; should investigate
    better code reuse.
    """

    def __init__(self, context, expr, parent_table):
        self.context = context
        self.expr = expr
        self.parent_table = parent_table

        qroots = self.parent_table._root_tables()
        self.query_roots = util.IbisSet.from_list(qroots)

    def get_result(self):
        self.foreign_table = None
        self.predicates = []

        self._visit(self.expr)

        if type(self.expr.op()) == ops.Any:
            op = ExistsSubquery(self.foreign_table, self.predicates)
        else:
            op = NotExistsSubquery(self.foreign_table, self.predicates)

        expr_type = dt.boolean.array_type()
        return expr_type(op)

    def _visit(self, expr):
        node = expr.op()

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg)
            elif isinstance(arg, ir.BooleanColumn):
                for sub_expr in L.flatten_predicate(arg):
                    self.predicates.append(sub_expr)
                    self._visit(sub_expr)
            elif isinstance(arg, ir.Expr):
                self._visit(arg)
            else:
                continue

    def _visit_table(self, expr):
        node = expr.op()

        if isinstance(expr, ir.TableExpr):
            base_table = _find_blocking_table(expr)
            if base_table is not None:
                base_node = base_table.op()
                if self._is_root(base_node):
                    pass
                else:
                    # Foreign ref
                    self.foreign_table = expr
        else:
            if not node.blocks():
                for arg in node.flat_args():
                    if isinstance(arg, ir.Expr):
                        self._visit(arg)

    def _is_root(self, what):
        if isinstance(what, ir.Expr):
            what = what.op()
        return what in self.query_roots


def _find_blocking_table(expr):
    node = expr.op()

    if node.blocks():
        return expr

    for arg in node.flat_args():
        if isinstance(arg, ir.Expr):
            result = _find_blocking_table(arg)
            if result is not None:
                return result
