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
        self.subexprs = {}
        self.visit_memo = set()

    def __contains__(self, obj):
        return self._key(obj) in self.formatted

    def _key(self, expr):
        memo_key = id(expr)
        if memo_key in self._repr_memo:
            return self._repr_memo[memo_key]

        result = self._format(expr)
        self._repr_memo[memo_key] = result

        return result

    def _format(self, expr):
        return expr.op()._repr(memo=self)

    def observe(self, expr, formatter=None):
        if formatter is None:
            formatter = self._format
        key = self._key(expr)
        if key not in self.formatted:
            self.aliases[key] = 'ref_%d' % len(self.formatted)
            self.formatted[key] = formatter(expr)
            self.ops[key] = expr.op()

        self.counts[key] += 1

    def count(self, expr):
        return self.counts[self._key(expr)]

    def get_alias(self, expr):
        return self.aliases[self._key(expr)]

    def get_formatted(self, expr):
        return self.formatted[self._key(expr)]


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
        if memo is None:
            memo = FormatMemo()

        self.memo = memo

    def get_result(self):
        what = self.expr.op()

        if self.memoize:
            self._memoize_tables()

        if isinstance(what, ops.TableNode) and what.has_schema():
            # This should also catch aggregations
            if not self.memoize and self.expr in self.memo:
                text = 'Table: %s' % self.memo.get_alias(self.expr)
            elif isinstance(what, ops.PhysicalTable):
                text = self._format_table(self.expr)
            else:
                # Any other node type
                text = self._format_node(self.expr)
        elif isinstance(what, ops.TableColumn):
            text = self._format_column(self.expr)
        elif isinstance(what, ir.Literal):
            text = 'Literal[{}]\n  {}'.format(
                self._get_type_display(), str(what.value)
            )
        elif isinstance(what, ir.ScalarParameter):
            text = 'ScalarParameter[{}]'.format(self._get_type_display())
        elif isinstance(what, ir.Node):
            text = self._format_node(self.expr)

        if isinstance(self.expr, ir.ValueExpr) and self.expr._name is not None:
            text = '{} = {}'.format(self.expr.get_name(), text)

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
        if id(self.expr) in self.memo.visit_memo:
            return

        stack = [self.expr]
        seen = set()
        memo = self.memo

        while stack:
            e = stack.pop()
            op = e.op()

            if op not in seen:
                seen.add(op)

                if isinstance(op, ops.PhysicalTable):
                    memo.observe(e, self._format_table)
                elif isinstance(op, ir.Node):
                    stack.extend(
                        arg for arg in reversed(op.args)
                        if isinstance(arg, ir.Expr)
                    )
                    if isinstance(op, table_memo_ops):
                        memo.observe(e, self._format_node)
                elif isinstance(op, ops.TableNode) and op.has_schema():
                    memo.observe(e, self._format_table)

                memo.visit_memo.add(id(e))

    def _indent(self, text, indents=1):
        return util.indent(text, self.indent_size * indents)

    def _format_table(self, expr):
        table = expr.op()
        # format the schema
        rows = ['name: {0!s}\nschema:'.format(table.name)]
        rows.extend(['  %s : %s' % tup for tup in
                     zip(table.schema.names, table.schema.types)])
        opname = type(table).__name__
        type_display = self._get_type_display(expr)
        opline = '%s[%s]' % (opname, type_display)
        return '{0}\n{1}'.format(opline, self._indent('\n'.join(rows)))

    def _format_column(self, expr):
        # HACK: if column is pulled from a Filter of another table, this parent
        # will not be found in the memo
        col = expr.op()
        parent = col.parent()

        if parent not in self.memo:
            self.memo.observe(parent, formatter=self._format_node)

        table_formatted = self.memo.get_alias(parent)
        table_formatted = self._indent(table_formatted)

        type_display = self._get_type_display(self.expr)
        return ("Column[{0}] '{1}' from table\n{2}"
                .format(type_display, col.name, table_formatted))

    def _format_node(self, expr):
        op = expr.op()
        formatted_args = []

        def visit(what, extra_indents=0):
            if isinstance(what, ir.Expr):
                result = self._format_subexpr(what)
            else:
                result = self._indent(str(what))

            if extra_indents > 0:
                result = util.indent(result, self.indent_size)

            formatted_args.append(result)

        arg_names = op._arg_names

        if not arg_names:
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
        type_display = self._get_type_display(expr)
        opline = '%s[%s]' % (opname, type_display)
        return '\n'.join([opline] + formatted_args)

    def _format_subexpr(self, expr):
        key = id(expr)
        if key not in self.memo.subexprs:
            formatter = ExprFormatter(expr, memo=self.memo, memoize=False)
            self.memo.subexprs[key] = self._indent(formatter.get_result(), 1)

        return self.memo.subexprs[key]

    def _get_type_display(self, expr=None):
        if expr is None:
            expr = self.expr

        return expr._type_display()
