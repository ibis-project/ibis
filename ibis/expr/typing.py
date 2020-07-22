""" Define types for annotation """

from typing import Optional, Tuple, Union

import pandas as pd

# Time context types
SupportsTimestamp = Union[pd.Timestamp, str]
TimeContext = Optional[Tuple[SupportsTimestamp, SupportsTimestamp]]
