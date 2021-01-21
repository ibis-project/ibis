from typing import Callable, Dict, List, Tuple, Type

import dask.dataframe as dd
from dask.dataframe.groupby import SeriesGroupBy

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


def make_selected_obj(gs: SeriesGroupBy):
    """
    When you select a column from a `pandas.DataFrameGroupBy` the underlying
    `.obj` reflects that selection. This function emulates that behavior.
    """
    # TODO profile this for data shuffling
    # We specify drop=False in the case that we are grouping on the column
    # we are selecting
    if isinstance(gs.obj, dd.Series):
        return gs.obj
    else:
        return gs.obj.set_index(gs.index, drop=False)[
            gs._meta._selected_obj.name
        ]
