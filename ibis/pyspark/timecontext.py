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
    timecontext: TimeCo ntext

    Returns
    -------
    filtered Spark Dataframe
    """
    if not timecontext:
        return df

    begin, end = timecontext
    if TIME_COL in df.columns:
        return df.filter((F.col(TIME_COL) >= begin) & (F.col(TIME_COL) < end))
    else:
        raise com.TranslationError(
            "'time' column missing in Dataframe {}."
            "To use time context, a Timestamp column name 'time' must"
            "present in the table. ".format(df)
        )


def union_time_context(
    timecontexts: List[TimeContext],
) -> Optional[TimeContext]:
    """ Return a 'union' time context of `timecontexts`

    Parameters
    ----------
    timecontexts: List[TimeContext]

    Returns
    -------
    TimeContext, a time context that start from the earliest begin time
    of `timecontexts`, and end with the latest end time of `timecontexts`.
    This is not a union in mathematical way.
    """
    begin = min([t[0] for t in timecontexts if t], default=None)
    end = max([t[1] for t in timecontexts if t], default=None)
    if begin and end:
        return begin, end
    return None
