"""
Aggregation functions for the arrow backend.
"""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AggregationContext:
    """Defines an abstract aggregation context."""
    __slots__ = 'parent', 'group_by', 'order_by', 'dtype'

    def __init__(self, parent=None, group_by=None, order_by=None, dtype=None):
        self.parent = parent
        self.group_by = group_by
        self.order_by = order_by
        self.dtype = dtype

    @abc.abstractmethod
    def agg(self, grouped_data, function, *args, **kwargs):
        """Abstract method to further perform the aggregation."""
        pass


def _apply(function, args, kwargs):
    assert callable(function), 'function {} is not callable'.format(function)
    return lambda data, function=function, args=args, kwargs=kwargs: (
        function(data, *args, **kwargs)
    )


class Summarize(AggregationContext):
    """Aggregation summarizing a column into one value."""
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if isinstance(function, six.string_types):
            return getattr(grouped_data, function)(*args, **kwargs)

        if not callable(function):
            raise TypeError(
                'Object {} is not callable or a string'.format(function)
            )

        return grouped_data.apply(_apply(function, args, kwargs))
