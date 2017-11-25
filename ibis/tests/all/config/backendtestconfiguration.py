import abc

try:
    from abc import abstractclassmethod
except ImportError:
    from abc import abstractmethod as abstractclassmethod

import six

import pandas.util.testing as tm


class BackendTestConfiguration(six.with_metaclass(abc.ABCMeta)):
    check_dtype = True
    check_names = True
    check_index = True

    @abstractclassmethod
    def connect(cls, backend):
        pass

    @classmethod
    def assert_series_equal(cls, *args, **kwargs):
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        kwargs.setdefault('check_index', cls.check_index)
        return tm.assert_series_equal(*args, **kwargs)
