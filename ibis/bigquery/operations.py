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


class DateDiff(_ops.ValueOp):

    input_type = [rules.date,
                  rules.date,
                  rules.string_options(
                      ['DAY', 'MONTH', 'QUARTER', 'YEAR'],
                      case_sensitive=False,
                      name='date_part',
                      default='DAY',
                      optional=True,
                  ),
                  ]
    output_type = rules.shape_like_arg(0, 'int64')


class TimestampDiff(_ops.ValueOp):

    input_type = [rules.timestamp,
                  rules.timestamp,
                  rules.string_options(
                      ['MICROSECOND', 'MILLISECOND', 'SECOND',
                       'MINUTE', 'HOUR', 'DAY', 'MONTH', 'QUARTER', 'YEAR'],
                      case_sensitive=False,
                      name='date_part',
                      default='SECOND',
                      optional=True,
                  ),
                  ]
    output_type = rules.shape_like_arg(0, 'int64')


class TimeDiff(_ops.ValueOp):

    input_type = [rules.time,
                  rules.time,
                  rules.string_options(
                      ['MICROSECOND', 'MILLISECOND', 'SECOND',
                       'MINUTE', 'HOUR'],
                      case_sensitive=False,
                      name='part',
                      default='SECOND',
                      optional=True,
                  ),
                  ]
    output_type = rules.shape_like_arg(0, 'int64')
