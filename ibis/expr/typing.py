""" Define types for annotation """

from typing import Tuple

import pandas as pd

# Time context types
TimeContext = Tuple[pd.Timestamp, pd.Timestamp]
