from __future__ import annotations

import ibis


def test_uuid():
    u = ibis.uuid()
    assert u.type().is_uuid()
    assert isinstance(u.op().shape, ibis.expr.datashape.Scalar)
