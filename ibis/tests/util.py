"""Test utilities."""

import ibis
import ibis.util as util


def assert_equal(left, right):
    """Assert that two ibis objects are equal."""

    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right), 'Comparing schemas: \n{!r} !=\n{!r}'.format(
            left, right
        )
    else:
        assert left.equals(right), 'Objects unequal: {}\nvs\n{}'.format(
            repr(left), repr(right)
        )


def random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


class TempHelper:
    def __init__(
        self, con, kind, create=False, create_kwargs=dict(), method_name=None,
    ):
        self.name = random_identifier(kind)
        self.con = con
        self.kind = kind
        self.create = create
        self.create_kwargs = create_kwargs
        self.method_name = kind if method_name is None else method_name

    def __enter__(self):
        # some of the temp entities may not support 'force' parameter
        # at 'drop' method, that's why we use 'try-except' for that
        try:
            getattr(self.con, 'drop_' + self.method_name)(self.name)
        except Exception:
            pass

        if self.create:
            getattr(self.con, 'create_' + self.method_name)(
                self.name, **self.create_kwargs
            )
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            getattr(self.con, 'drop_' + self.method_name)(self.name)
        except Exception:
            pass
