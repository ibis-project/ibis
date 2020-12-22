import pickle
from collections import OrderedDict

import ibis


def test_pickle_struct_value():
    struct_scalar_expr = ibis.literal(
        OrderedDict([("fruit", "pear"), ("weight", 0)])
    )

    raw = pickle.dumps(struct_scalar_expr)
    loaded = pickle.loads(raw)

    assert loaded.equals(struct_scalar_expr)
