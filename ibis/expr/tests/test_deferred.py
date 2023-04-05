import pytest

import ibis


def test_invalid_attribute():
    from ibis import _

    t = ibis.table(dict(a="int"), name="t")
    d = _.a + _.b
    with pytest.raises(AttributeError):
        d.resolve(t)
