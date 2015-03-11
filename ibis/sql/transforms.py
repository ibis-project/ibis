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


import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.expr.types as ir


class ExistsSubquery(ir.Node):

    """
    Helper class
    """

    def __init__(self, foreign_table, predicates):
        self.foreign_table = foreign_table
        self.predicates = predicates
        ir.Node.__init__(self, [foreign_table, predicates])


class NotExistsSubquery(ir.Node):

    def __init__(self, foreign_table, predicates):
        self.foreign_table = foreign_table
        self.predicates = predicates
        ir.Node.__init__(self, [foreign_table, predicates])


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
        self.query_roots = set([id(x) for x in qroots])

    def get_result(self):
        self.foreign_table = None
        self.predicates = []

        self._visit(self.expr)

        if type(self.expr.op()) == ops.Any:
            op = ExistsSubquery(self.foreign_table, self.predicates)
        else:
            op = NotExistsSubquery(self.foreign_table, self.predicates)

        return ir.BooleanArray(op)

    def _visit(self, expr):
        node = expr.op()

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg)
            elif isinstance(arg, ir.BooleanArray):
                for sub_expr in L.unwrap_ands(arg):
                    self.predicates.append(sub_expr)
                    self._visit(sub_expr)
            elif isinstance(arg, ir.Expr):
                self._visit(arg)
            else:
                continue

    def _visit_table(self, expr):
        node = expr.op()

        if isinstance(node, (ops.PhysicalTable, ops.SelfReference)):
            self._ref_check(expr)

        for arg in node.flat_args():
            if isinstance(arg, ir.Expr):
                self._visit(arg)

    def _ref_check(self, expr):
        node = expr.op()

        if self._is_root(node):
            pass
        else:
            # Foreign ref
            if isinstance(node, ops.SelfReference):
                foreign_table = node.table
            else:
                foreign_table = expr

            self.foreign_table = foreign_table

    def _is_root(self, what):
        if isinstance(what, ir.Expr):
            what = what.op()
        return id(what) in self.query_roots
