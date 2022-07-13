from ibis.common.validators import container_of
from ibis.expr import rules as rlz
from ibis.expr.operations.tabular import TabularUserDefinedFunction

from .vectorized import UserDefinedFunction, _coerce_to_dataframe


class UserDefinedTable(UserDefinedFunction):
    """Class representing a user defined table.

    This class Implements __call__ that returns an ibis expr for the UDT.
    """

    input_type = container_of(rlz.datatype, type=tuple, max_length=0)

    def __init__(self, func, func_type, output_type):
        super().__init__(func, func_type, [], output_type)

    def _get_coercion_function(self):
        """Returns the function to coerce the result to a dataframe."""
        return _coerce_to_dataframe


def _udt_decorator(node_type, output_type):
    def wrapper(func):
        return UserDefinedTable(func, node_type, output_type)

    return wrapper


def tabular(output_type):
    return _udt_decorator(TabularUserDefinedFunction, output_type)
