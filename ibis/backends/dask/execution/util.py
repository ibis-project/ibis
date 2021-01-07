from typing import Callable, Dict, List, Tuple, Type

import ibis.expr.operations as ops
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher

DispatchRule = Tuple[Tuple[Type], Callable]

TypeRegistrationDict = Dict[ops.Node, List[DispatchRule]]


def register_types_to_dispatcher(
    dispatcher: TraceTwoLevelDispatcher, types: TypeRegistrationDict
):
    """
    Many dask operations utilize the functions defined in the pandas backend
    without modification. This function helps perform registrations in bulk
    """
    for ibis_op, registration_list in types.items():
        for types_to_register, fn in registration_list:
            dispatcher.register(ibis_op, *types_to_register)(fn)
