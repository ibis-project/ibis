from io import StringIO

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.util as util
from ibis.backends.base.sql.compiler import (
    Compiler,
    ExprTranslator,
    Select,
    SelectBuilder,
    TableSetFormatter,
)

from .identifiers import quote_identifier
from .registry import operation_registry


class ClickhouseSelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class ClickhouseSelect(Select):
    def format_group_by(self):
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            columns = [
                '`{0}`'.format(expr.get_name()) for expr in self.group_by
            ]
            clause = 'GROUP BY {0}'.format(', '.join(columns))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {0}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_limit(self):
        if not self.limit:
            return None

        buf = StringIO()

        n, offset = self.limit['n'], self.limit['offset']
        buf.write('LIMIT {}'.format(n))
        if offset is not None and offset != 0:
            buf.write(', {}'.format(offset))

        return buf.getvalue()


class ClickhouseTableSetFormatter(TableSetFormatter):

    _join_names = {
        ops.InnerJoin: 'ALL INNER JOIN',
        ops.LeftJoin: 'ALL LEFT JOIN',
        ops.AnyInnerJoin: 'ANY INNER JOIN',
        ops.AnyLeftJoin: 'ANY LEFT JOIN',
    }

    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        # TODO: Now actually format the things
        buf = StringIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(
            self.join_types, self.join_tables[1:], self.join_predicates
        ):
            buf.write('\n')
            buf.write(util.indent('{0} {1}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = map(self._format_predicate, preds)
                fmt_preds = util.indent(
                    'USING ' + ', '.join(fmt_preds), self.indent * 2
                )
                buf.write(fmt_preds)

        return buf.getvalue()

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()
            if not isinstance(op, ops.Equals):
                raise com.TranslationError(
                    'Non-equality join predicates are ' 'not supported'
                )

            left_on, right_on = op.args
            if left_on.get_name() != right_on.get_name():
                raise com.TranslationError(
                    'Joining on different column names ' 'is not supported'
                )

    def _format_predicate(self, predicate):
        column = predicate.op().args[0]
        return quote_identifier(column.get_name(), force=True)

    def _quote_identifier(self, name):
        return quote_identifier(name)


class ClickhouseExprTranslator(ExprTranslator):
    _registry = operation_registry


rewrites = ClickhouseExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


class ClickhouseCompiler(Compiler):
    translator_class = ClickhouseExprTranslator
    table_set_formatter_class = ClickhouseTableSetFormatter
    select_builder_class = ClickhouseSelectBuilder
    select_class = ClickhouseSelect
