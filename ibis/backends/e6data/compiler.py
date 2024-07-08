from __future__ import annotations

import string
from public import public

from ibis.backends.mysql.compiler import MySQLCompiler
from ibis.backends.sql.datatypes import E6DataType
from ibis.backends.sql.dialects import E6data
from ibis.common.patterns import replace
from ibis.expr.rewrites import p


@replace(p.Limit)
def rewrite_limit(_, **kwargs):
    """Rewrite limit for MySQL to include a large upper bound.

    From the MySQL docs @ https://dev.mysql.com/doc/refman/8.0/en/select.html

    > To retrieve all rows from a certain offset up to the end of the result
    > set, you can use some large number for the second parameter. This statement
    > retrieves all rows from the 96th row to the last:
    >
    > SELECT * FROM tbl LIMIT 95,18446744073709551615;
    """
    if _.n is None and _.offset is not None:
        some_large_number = (1 << 64) - 1
        return _.copy(n=some_large_number)
    return _


@public
class E6DataCompiler(MySQLCompiler):
    __slots__ = ()

    dialect = E6data
    type_mapper = E6DataType
