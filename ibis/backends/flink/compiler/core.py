"""Flink ibis expression to SQL string compiler."""

from __future__ import annotations

import functools

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import DDL, DML, Compiler, Select, SelectBuilder
from ibis.backends.flink.translator import FlinkExprTranslator


class FlinkSelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class FlinkSelect(Select):
    def format_group_by(self) -> str:
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            group_keys = map(self._translate, self.group_by)
            clause = 'GROUP BY {}'.format(', '.join(list(group_keys)))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)


class FlinkCompiler(Compiler):
    select_builder_class = FlinkSelectBuilder
    select_class = FlinkSelect
    cheap_in_memory_tables = True
    translator_class = FlinkExprTranslator


def translate(op: ops.TableNode) -> str:
    # TODO(chloeh13q): support translation of non-select exprs (e.g. literals)
    ast = FlinkCompiler.to_ast(op)
    return translate_language(ast.queries[0])


@functools.singledispatch
def translate_language(language: DML | DDL) -> str:
    raise com.OperationNotDefinedError(f'No translation rule for {type(language)}')


@translate_language.register(Select)
def _select(language: Select):
    return language.compile()
