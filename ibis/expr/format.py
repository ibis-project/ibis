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

import ibis.util as util

import ibis.expr.types as ir
import ibis.expr.operations as ops


class FormatMemo(object):
    # A little sanity hack to simplify the below

    def __init__(self):
        from collections import defaultdict
        self.formatted = {}
        self.aliases = {}
        self.ops = {}
        self.counts = defaultdict(lambda: 0)
        self._repr_memo = {}

    def __contains__(self, obj):
        return self._key(obj) in self.formatted

    def _key(self, obj):
        memo_key = id(obj)
        if memo_key in self._repr_memo:
            return self._repr_memo[memo_key]
        result = self._format(obj)
        self._repr_memo[memo_key] = result

        return result

    def _format(self, obj):
        return obj._repr(memo=self._repr_memo)

    def observe(self, obj, formatter=lambda x: x._repr()):
        key = self._key(obj)
        if key not in self.formatted:
            self.aliases[key] = 'ref_%d' % len(self.formatted)
            self.formatted[key] = formatter(obj)
            self.ops[key] = obj

        self.counts[key] += 1

    def count(self, obj):
        return self.counts[self._key(obj)]

    def get_alias(self, obj):
        return self.aliases[self._key(obj)]

    def get_formatted(self, obj):
        return self.formatted[self._key(obj)]


class ExprFormatter(object):

    """
    For creating a nice tree-like representation of an expression graph for
    displaying in the console.

    TODO: detect reused DAG nodes and do not display redundant information
    """

    def __init__(self, expr, indent_size=2, base_level=0, memo=None,
                 memoize=True):
        self.expr = expr
        self.indent_size = indent_size
        self.base_level = base_level

        self.memoize = memoize

        # For tracking "extracted" objects, like tables, that we don't want to
        # print out more than once, and simply alias in the expression tree
        self.memo = memo or FormatMemo()

    def get_result(self):
        what = self.expr.op()

        if self.memoize:
            self._memoize_tables()

        if isinstance(what, ops.TableNode) and what.has_schema():
            # This should also catch aggregations
            if not self.memoize and what in self.memo:
                text = 'Table: %s' % self.memo.get_alias(what)
            elif isinstance(what, ops.PhysicalTable):
                text = self._format_table(what)
            else:
                # Any other node type
                text = self._format_node(what)
        elif isinstance(what, ops.TableColumn):
            text = self._format_column(self.expr)
        elif isinstance(what, ir.Node):
            text = self._format_node(what)
        elif isinstance(what, ops.Literal):
            text = 'Literal[%s] %s' % (self._get_type_display(),
                                       str(what.value))

        if isinstance(self.expr, ir.ValueExpr) and self.expr._name is not None:
            text = '{0} = {1}'.format(self.expr.get_name(), text)

        if self.memoize:
            alias_to_text = [(self.memo.aliases[x],
                              self.memo.formatted[x],
                              self.memo.ops[x])
                             for x in self.memo.formatted]
            alias_to_text.sort()

            # A hack to suppress printing out of a ref that is the result of
            # the top level expression
            refs = [x + '\n' + y
                    for x, y, op in alias_to_text
                    if not op.equals(what)]

            text = '\n\n'.join(refs + [text])

        return self._indent(text, self.base_level)

    def _memoize_tables(self):
        table_memo_ops = (ops.Aggregation, ops.Selection,
                          ops.SelfReference)

        def walk(expr):
            op = expr.op()

            def visit(arg):
                if isinstance(arg, list):
                    [visit(x) for x in arg]
                elif isinstance(arg, ir.Expr):
                    walk(arg)

            if isinstance(op, ops.PhysicalTable):
                self.memo.observe(op, self._format_table)
            elif isinstance(op, ir.Node):
                visit(op.args)
                if isinstance(op, table_memo_ops):
                    self.memo.observe(op, self._format_node)
            elif isinstance(op, ops.TableNode) and op.has_schema():
                self.memo.observe(op, self._format_table)

        walk(self.expr)

    def _indent(self, text, indents=1):
        return util.indent(text, self.indent_size * indents)

    def _format_table(self, table):
        # format the schema
        rows = ['name: {0!s}\nschema:'.format(table.name)]
        rows.extend(['  %s : %s' % tup for tup in
                     zip(table.schema.names, table.schema.types)])
        opname = type(table).__name__
        type_display = self._get_type_display(table)
        opline = '%s[%s]' % (opname, type_display)
        return '{0}\n{1}'.format(opline, self._indent('\n'.join(rows)))

    def _format_column(self, expr):
        # HACK: if column is pulled from a Filter of another table, this parent
        # will not be found in the memo
        col = expr.op()
        parent_op = col.parent().op()
        if parent_op in self.memo:
            table_formatted = self.memo.get_alias(parent_op)
        else:
            table_formatted = '\n' + self._indent(self._format_node(parent_op))
        type_display = self._get_type_display(self.expr)
        return ("Column[{0}] '{1}' from table {2}"
                .format(type_display, col.name, table_formatted))

    def _format_node(self, op):
        formatted_args = []

        def visit(what, extra_indents=0):
            if isinstance(what, ir.Expr):
                result = self._format_subexpr(what)
            else:
                result = self._indent(str(what))

            if extra_indents > 0:
                result = util.indent(result, self.indent_size)

            formatted_args.append(result)

        arg_names = getattr(op, '_arg_names', None)

        if arg_names is None:
            for arg in op.args:
                if isinstance(arg, list):
                    for x in arg:
                        visit(x)
                else:
                    visit(arg)
        else:
            for arg, name in zip(op.args, arg_names):
                if name is not None:
                    name = self._indent('{0}:'.format(name))
                if isinstance(arg, list):
                    if name is not None and len(arg) > 0:
                        formatted_args.append(name)
                        indents = 1
                    else:
                        indents = 0
                    for x in arg:
                        visit(x, extra_indents=indents)
                else:
                    if name is not None:
                        formatted_args.append(name)
                        indents = 1
                    else:
                        indents = 0
                    visit(arg, extra_indents=indents)

        opname = type(op).__name__
        type_display = self._get_type_display(op)
        opline = '%s[%s]' % (opname, type_display)

        return '\n'.join([opline] + formatted_args)

    def _format_subexpr(self, expr):
        formatter = ExprFormatter(expr, base_level=1, memo=self.memo,
                                  memoize=False)
        return formatter.get_result()

    def _get_type_display(self, expr=None):
        if expr is None:
            expr = self.expr

        if isinstance(expr, ir.Node):
            expr = expr.to_expr()

        if isinstance(expr, ir.TableExpr):
            return 'table'
        elif isinstance(expr, ir.ArrayExpr):
            return 'array(%s)' % expr.type()
        elif isinstance(expr, ir.SortExpr):
            return 'array-sort'
        elif isinstance(expr, (ir.ScalarExpr, ir.AnalyticExpr)):
            return '%s' % expr.type()
        elif isinstance(expr, ir.ExprList):
            list_args = [self._get_type_display(arg)
                         for arg in expr.op().args]
            return ', '.join(list_args)
        else:
            raise NotImplementedError
