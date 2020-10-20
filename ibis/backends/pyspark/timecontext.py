from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

import ibis.common.exceptions as com
from ibis.expr.timecontext import TIME_COL
from ibis.expr.typing import TimeContext


def filter_by_time_context(
    df: DataFrame, timecontext: Optional[TimeContext] = None
) -> DataFrame:
    """ Filter a Dataframe by given time context
    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
    timecontext: TimeContext

    Returns
    -------
    filtered Spark Dataframe
    """
    if not timecontext:
        return df

    if TIME_COL in df.columns:
        # For py3.8, underlying spark type converter calls utctimetuple()
        # and will throw excpetion for Timestamp type if tz is set.
        # See https://github.com/pandas-dev/pandas/issues/32174
        # Dropping tz will cause spark to interpret begin, end with session
        # timezone & os env TZ. We convert Timestamp to pydatetime to
        # workaround.
        begin, end = timecontext
        return df.filter(
            (F.col(TIME_COL) >= begin.to_pydatetime())
            & (F.col(TIME_COL) < end.to_pydatetime())
        )
    else:
        raise com.TranslationError(
            "'time' column missing in Dataframe {}."
            "To use time context, a Timestamp column name 'time' must"
            "present in the table. ".format(df)
        )


def combine_time_context(
    timecontexts: List[TimeContext],
) -> Optional[TimeContext]:
    """ Return a combined time context of `timecontexts`

    The combined time context starts from the earliest begin time
    of `timecontexts`, and ends with the latest end time of `timecontexts`
    The motivation is to generate a time context that is a superset
    to all time contexts.

    Examples
    ---------
    >>> timecontexts = [(pd.Timestamp('20200102'), pd.Timestamp('20200103')),
        (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
        (pd.Timestamp('20200109'), pd.Timestamp('20200110')),
    ]
    >>> combine_time_context(timecontexts)
    (pd.Timestamp('20200101'), pd.Timestamp('20200110'))

    >>> timecontexts = [None]
    >>> combine_time_context(timecontexts)
    None

    Parameters
    ----------
    timecontexts: List[TimeContext]

    Returns
    -------
    TimeContext
    """
    begin = min([t[0] for t in timecontexts if t], default=None)
    end = max([t[1] for t in timecontexts if t], default=None)
    if begin and end:
        return begin, end
    return None
