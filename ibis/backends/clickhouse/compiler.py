from io import StringIO

import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    Compiler,
    ExprTranslator,
    Select,
    SelectBuilder,
    TableSetFormatter,
)

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
            columns = [f'`{expr.get_name()}`' for expr in self.group_by]
            clause = 'GROUP BY {}'.format(', '.join(columns))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_limit(self):
        if not self.limit:
            return None

        buf = StringIO()

        n, offset = self.limit['n'], self.limit['offset']
        if offset is not None and offset != 0:
            buf.write(f'LIMIT {offset}, {n}')
        else:
            buf.write(f'LIMIT {n}')

        return buf.getvalue()


class ClickhouseTableSetFormatter(TableSetFormatter):

    _join_names = {
        ops.InnerJoin: 'ALL INNER JOIN',
        ops.LeftJoin: 'ALL LEFT JOIN',
        ops.AnyInnerJoin: 'ANY INNER JOIN',
        ops.AnyLeftJoin: 'ANY LEFT JOIN',
    }

    _non_equijoin_supported = False


class ClickhouseExprTranslator(ExprTranslator):
    _registry = operation_registry


rewrites = ClickhouseExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


@rewrites(ops.DayOfWeekName)
def day_of_week_name(expr):
    # ClickHouse 20 doesn't support dateName
    #
    # ClickHouse 21 supports dateName is broken for regexen:
    # https://github.com/ClickHouse/ClickHouse/issues/32777
    #
    # ClickHouses 20 and 21 also have a broken case statement hence the ifnull:
    # https://github.com/ClickHouse/ClickHouse/issues/32849
    #
    # We test against 20 in CI, so we implement day_of_week_name as follows
    return (
        expr.op()
        .arg.day_of_week.index()
        .case()
        .when(0, "Monday")
        .when(1, "Tuesday")
        .when(2, "Wednesday")
        .when(3, "Thursday")
        .when(4, "Friday")
        .when(5, "Saturday")
        .when(6, "Sunday")
        .else_("")
        .end()
        .nullif("")
    )


class ClickhouseCompiler(Compiler):
    translator_class = ClickhouseExprTranslator
    table_set_formatter_class = ClickhouseTableSetFormatter
    select_builder_class = ClickhouseSelectBuilder
    select_class = ClickhouseSelect
