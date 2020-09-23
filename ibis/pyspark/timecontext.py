from typing import Optional

import pyspark.sql.functions as F

import ibis.common.exceptions as com
from ibis.expr.timecontext import TIME_COL
from ibis.expr.typing import TimeContext


def filter_by_time_context(df, timecontext: Optional[TimeContext] = None):
    """ Filter a Dataframe by given time context
    Parameters
    ----------
    df : Spark DataFrame
    timecontext: TimeContext

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
