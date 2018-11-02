from __future__ import absolute_import

import functools
import itertools
import unittest.mock as mock

from contextlib import suppress
from decimal import Decimal
from functools import reduce
from inspect import signature, Parameter
from pkg_resources import parse_version  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np

from io import BytesIO, StringIO, string_types  # noqa: F401

zip = zip
zip_longest = itertools.zip_longest
range = range
map = map
import builtins
import pickle
maketrans = str.maketrans


integer_types = int, np.integer


# pandas compat
try:
    from pandas.api.types import (  # noqa: F401
        DatetimeTZDtype, CategoricalDtype, infer_dtype
    )
except ImportError:
    from pandas.types.dtypes import (  # noqa: F401
        DatetimeTZDtype, CategoricalDtype, infer_dtype
    )

try:
    from pandas.core.tools.datetimes import to_time, to_datetime  # noqa: F401
except ImportError:
    from pandas.tseries.tools import to_time, to_datetime  # noqa: F401


def to_date(*args, **kwargs):
    return to_datetime(*args, **kwargs).date()
