# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import itertools

import numpy as np

import sys
import six
from six import BytesIO, StringIO, string_types  # noqa: F401


PY2 = sys.version_info[0] == 2


if not PY2:
    unicode_type = str

    def lzip(*x):
        return list(zip(*x))

    zip = zip
    zip_longest = itertools.zip_longest

    def dict_values(x):
        return list(x.values())

    from decimal import Decimal
    import unittest.mock as mock
    range = range
    import builtins
    import pickle
    maketrans = str.maketrans
    from functools import reduce
else:
    try:
        from cdecimal import Decimal
    except ImportError:
        from decimal import Decimal  # noqa: F401

    unicode_type = unicode  # noqa: F821
    lzip = zip
    zip = itertools.izip
    zip_longest = itertools.izip_longest

    def dict_values(x):
        return x.values()

    try:
        import mock  # noqa: F401
    except ImportError:
        pass

    import __builtin__ as builtins  # noqa: F401

    range = xrange  # noqa: F821
    reduce = reduce  # noqa: F821
    import cPickle as pickle  # noqa: F401

    def maketrans(x, y):
        import string

        assert type(x) == type(y), 'type(x) != type(y) -> {} != {}'.format(
            type(x), type(y)
        )
        assert len(x) == len(y), 'len(x) != len(y) -> {:d} != {:d}'.format(
            len(x), len(y)
        )

        if isinstance(x, six.text_type):
            return dict(zip(map(ord, x), y))
        return string.maketrans(x, y)

    reduce = reduce


integer_types = six.integer_types + (np.integer,)


try:
    from pathlib import Path  # noqa: F401
except ImportError:
    # py2 compat
    from pathlib2 import Path  # noqa: F401


try:
    from contextlib import suppress
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def suppress(*exceptions):
        try:
            yield
        except exceptions:
            pass


try:
    from pkg_resources import parse_version  # noqa: F401
except ImportError:
    from distutils.version import LooseVersion as parse_version  # noqa: F401


# pandas compat
try:
    from pandas.api.types import (DatetimeTZDtype,  # noqa: F401
                                  CategoricalDtype)  # noqa: F401
except ImportError:
    from pandas.types.dtypes import (DatetimeTZDtype,  # noqa: F401
                                     CategoricalDtype)  # noqa: F401

try:
    from pandas.core.tools.datetimes import to_time, to_datetime  # noqa: F401
except ImportError:
    from pandas.tseries.tools import to_time, to_datetime  # noqa: F401


def to_date(*args, **kwargs):
    return to_datetime(*args, **kwargs).date()


if PY2:
    def wrapped(f):
        def wrapper(functools_wrapped):
            functools_wrapped.__wrapped__ = f
            return functools_wrapped
        return wrapper
else:
    def wrapped(f):
        import toolz
        return toolz.identity
