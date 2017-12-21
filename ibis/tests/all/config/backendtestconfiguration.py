import abc

try:
    from abc import abstractclassmethod
except ImportError:
    from abc import abstractmethod as abstractclassmethod

import six

import pandas.util.testing as tm


class BackendTestConfiguration(six.with_metaclass(abc.ABCMeta)):
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = True
    check_names = True

    additional_skipped_operations = frozenset()

    @abstractclassmethod
    def connect(cls, backend):
        pass

    @classmethod
    def assert_series_equal(cls, *args, **kwargs):
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        return tm.assert_series_equal(*args, **kwargs)

    @classmethod
    def assert_frame_equal(cls, *args, **kwargs):
        return tm.assert_frame_equal(*args, **kwargs)

    @classmethod
    def default_series_rename(cls, series, name='tmp'):
        return series.rename(name)

    @classmethod
    def functional_alltypes(cls, con):
        return con.database().functional_alltypes


class UnorderedSeriesComparator(object):
    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        return super(UnorderedSeriesComparator, cls).assert_series_equal(
            left.sort_values().reset_index(drop=True),
            right.sort_values().reset_index(drop=True),
            *args,
            **kwargs
        )
