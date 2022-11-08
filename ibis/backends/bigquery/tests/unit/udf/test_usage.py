import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.bigquery import udf


def test_multiple_calls_redefinition():
    @udf([dt.string], dt.double)
    def my_len(s):
        return s.length

    s = ibis.literal("abcd")
    expr = my_len(s) + my_len(s)

    @udf([dt.string], dt.double)
    def my_len(s):
        return s.length + 1

    expr = expr + my_len(s)

    sql = ibis.bigquery.compile(expr)
    expected = '''\
CREATE TEMPORARY FUNCTION my_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return s.length;
}
return my_len(s);
""";

CREATE TEMPORARY FUNCTION my_len_1(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return (s.length + 1);
}
return my_len(s);
""";

SELECT (my_len_0('abcd') + my_len_0('abcd')) + my_len_1('abcd') AS `tmp`'''
    assert sql == expected


@pytest.mark.parametrize(
    ("argument_type", "return_type"),
    [
        param(
            dt.int64,
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="int_float",
        ),
        param(
            dt.float64,
            dt.int64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="float_int",
        ),
        # complex argument type, valid return type
        param(
            dt.Array(dt.int64),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="array_int_float",
        ),
        # valid argument type, complex invalid return type
        param(
            dt.float64,
            dt.Array(dt.int64),
            marks=pytest.mark.xfail(raises=TypeError),
            id="float_array_int",
        ),
        # both invalid
        param(
            dt.Array(dt.Array(dt.int64)),
            dt.int64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="array_array_int_int",
        ),
        # struct type with nested integer, valid return type
        param(
            dt.Struct.from_tuples([("x", dt.Array(dt.int64))]),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="struct",
        ),
    ],
)
def test_udf_int64(argument_type, return_type):
    # invalid argument type, valid return type
    @udf([argument_type], return_type)
    def my_int64_add(x):
        return 1.0
