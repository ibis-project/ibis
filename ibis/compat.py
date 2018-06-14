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

    def viewkeys(x):
        return x.keys()

    from decimal import Decimal
    from inspect import signature, Parameter, _empty
    import unittest.mock as mock
    range = range
    map = map
    import builtins
    import pickle
    maketrans = str.maketrans
    import functools
    from functools import reduce
else:
    try:
        from cdecimal import Decimal
    except ImportError:
        from decimal import Decimal  # noqa: F401

    from funcsigs import signature, Parameter, _empty  # noqa: F401

    unicode_type = unicode  # noqa: F821
    lzip = zip
    zip = itertools.izip
    zip_longest = itertools.izip_longest
    map = itertools.imap

    def viewkeys(x):
        return x.viewkeys()

    try:
        import mock  # noqa: F401
    except ImportError:
        pass

    import __builtin__ as builtins  # noqa: F401
    import functools32 as functools  # noqa: F401

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
