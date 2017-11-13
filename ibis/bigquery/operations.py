import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.operations as _ops
from ibis.expr.rules import value, number


class ApproxQuantile(_ops.Reduction):

    input_type = [value,
                  number(name='number', allow_boolean=False),
                  rules.boolean(name='distinct', default=False),
                  rules.boolean(name='ignore_nulls', default=True),
                  ]

    def output_type(self):
        return self.args[0].type().scalar_type()


class ApproxCountDistinct(_ops.Reduction):

    input_type = [value]

    def output_type(self):
        return self.args[0].type().scalar_type()


class FormatDate(_ops.ValueOp):

    input_type = [rules.date, rules.string]
    output_type = rules.shape_like_arg(0, 'string')
