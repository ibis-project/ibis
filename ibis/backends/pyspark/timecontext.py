from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as F

import ibis.common.exceptions as com
from ibis.backends.base.df.timecontext import TimeContext, get_time_col

if TYPE_CHECKING:
    from pyspark.sql.dataframe import DataFrame


def filter_by_time_context(
    df: DataFrame,
    timecontext: TimeContext | None,
    adjusted_timecontext: TimeContext | None = None,
) -> DataFrame:
    """Filter a Dataframe by given time context."""
    # Return original df if there is no timecontext (timecontext is not used)
    # or timecontext and adjusted_timecontext are the same
    if (not timecontext) or (
        timecontext and adjusted_timecontext and timecontext == adjusted_timecontext
    ):
        return df

    time_col = get_time_col()
    if time_col in df.columns:
        # For py3.8, underlying spark type converter calls utctimetuple()
        # and will throw exception for Timestamp type if tz is set.
        # See https://github.com/pandas-dev/pandas/issues/32174
        # Dropping tz will cause spark to interpret begin, end with session
        # timezone & os env TZ. We convert Timestamp to pydatetime to
        # workaround.
        begin, end = timecontext
        return df.filter(
            (F.col(time_col) >= begin.to_pydatetime())
            & (F.col(time_col) < end.to_pydatetime())
        )
    else:
        raise com.TranslationError(
            f"'time' column missing in Dataframe {df}."
            "To use time context, a Timestamp column name 'time' must"
            "present in the table. "
        )


def combine_time_context(
    timecontexts: list[TimeContext],
) -> TimeContext | None:
    """Return a combined time context of `timecontexts`.

    The combined time context starts from the earliest begin time
    of `timecontexts`, and ends with the latest end time of `timecontexts`
    The motivation is to generate a time context that is a superset
    to all time contexts.

    Examples
    --------
    >>> import pandas as pd
    >>> timecontexts = [
    ...     (pd.Timestamp("20200102"), pd.Timestamp("20200103")),
    ...     (pd.Timestamp("20200101"), pd.Timestamp("20200106")),
    ...     (pd.Timestamp("20200109"), pd.Timestamp("20200110")),
    ... ]
    >>> combine_time_context(timecontexts)
    (Timestamp(...), Timestamp(...))
    >>> timecontexts = [None]
    >>> print(combine_time_context(timecontexts))
    None
    """
    begin = min((t[0] for t in timecontexts if t), default=None)
    end = max((t[1] for t in timecontexts if t), default=None)
    if begin and end:
        return begin, end
    return None
