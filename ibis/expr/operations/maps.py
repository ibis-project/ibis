from public import public

from ...common import exceptions as com
from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
from .core import ValueOp


@public
class MapLength(ValueOp):
    arg = Arg(rlz.mapping)
    output_type = rlz.shape_like('arg', dt.int64)


@public
class MapValueForKey(ValueOp):
    arg = Arg(rlz.mapping)
    key = Arg(rlz.one_of([rlz.string, rlz.integer]))

    def output_type(self):
        return rlz.shape_like(tuple(self.args), self.arg.type().value_type)


@public
class MapValueOrDefaultForKey(ValueOp):
    arg = Arg(rlz.mapping)
    key = Arg(rlz.one_of([rlz.string, rlz.integer]))
    default = Arg(rlz.any)

    def output_type(self):
        arg = self.arg
        default = self.default
        map_type = arg.type()
        value_type = map_type.value_type
        default_type = default.type()

        if default is not None and not dt.same_kind(default_type, value_type):
            raise com.IbisTypeError(
                "Default value\n{}\nof type {} cannot be cast to map's value "
                "type {}".format(default, default_type, value_type)
            )

        result_type = dt.highest_precedence((default_type, value_type))
        return rlz.shape_like(tuple(self.args), result_type)


@public
class MapKeys(ValueOp):
    arg = Arg(rlz.mapping)

    def output_type(self):
        arg = self.arg
        return rlz.shape_like(arg, dt.Array(arg.type().key_type))


@public
class MapValues(ValueOp):
    arg = Arg(rlz.mapping)

    def output_type(self):
        arg = self.arg
        return rlz.shape_like(arg, dt.Array(arg.type().value_type))


@public
class MapConcat(ValueOp):
    left = Arg(rlz.mapping)
    right = Arg(rlz.mapping)
    output_type = rlz.typeof('left')
