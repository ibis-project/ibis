""" Time context adjustment algorithm
    In an Ibis tree, time context is local for each node, and they should be
    adjusted accordingly for some specific nodes. Those operations may
    require extra data outside of the global time context that user defines.
    For example, in asof_join, we need to look back extra `tolerance` daays
    for the right table to get the data for joining. Similarly for window
    operation with preceeding and following.
    Algorithm to calculate context adjustment are defined in this module
    and could be used by multiple backends.
"""
from ibis.expr.typing import TimeContext


def adjust_context_asof_join(
    timecontext: TimeContext, tolerance
) -> TimeContext:
    """
    Params
    -------
    timecontext: TimeContext, time context of the asof join node
    tolerance: pd.Timedelta, days to look back in joining

    Returns
    --------
    Adjusted time context for the right table in asof_join
    """
    begin, end = timecontext
    # only backwards and adjust begin time only
    return (begin - tolerance, end)


def adjust_context_window(
    timecontext: TimeContext, preceding=None, following=None
) -> TimeContext:
    """
    Params
    -------
    timecontext: TimeContext, time context of the window node
    preceding: Optional[pd.Timedelta], days to look back in window
    following: Optional[pd.Timedelta], days to look forward in window

    Returns
    --------
    Adjusted time context for window
    """

    # adjust time context by preceding and following
    begin, end = timecontext
    if preceding:
        begin = begin - preceding
    if following:
        end = end + following
    return (begin, end)
