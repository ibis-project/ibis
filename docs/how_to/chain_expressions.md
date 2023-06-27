# Chain expressions with the underscore API

Expressions can easily be chained using the deferred expression API, also known as the Underscore (`_`) API.

In this guide, we use the `_` API to concisely create column expressions and then chain table expressions.

## Setup

To get started, import `_` from ibis:

```python
import ibis
from ibis import _

import pandas as pd
```

Let's create two in-memory tables using [`ibis.memtable`](memtable_join.md), an API introduced in 3.2:

```python
t1 = ibis.memtable(pd.DataFrame({'x': range(5), 'y': list('ab')*2 + list('e')}))
t2 = ibis.memtable(pd.DataFrame({'x': range(10), 'z': list(reversed(list('ab')*2 + list('e')))*2}))
```

## Creating column expressions

We can use `_` to create new column expressions without explicit reference to the previous table expression:

```python
# We can pass a deferred expression into a function:
def modf(t):
    return t.x % 3

xmod = modf(_)

# We can create ColumnExprs like aggregate expressions:
ymax = _.y.max()
zmax = _.z.max()
zct = _.z.count()
```

## Chaining Ibis expressions

We can also use it to chain Ibis expressions in one Python expression:

```python
join = (
    t1
    # _ is t1
    .join(t2, _.x == t2.x)
    # _ is the join result:
    .mutate(xmod=xmod)
    # _ is the TableExpression after mutate:
    .group_by(_.xmod)
    # `ct` is a ColumnExpression derived from a deferred expression:
    .aggregate(ymax=ymax, zmax=zmax)
    # _ is the aggregation result:
    .filter(_.ymax == _.zmax)
    # _ is the filtered result, and re-create xmod in t2 using modf:
    .join(t2, _.xmod == modf(t2))
    # _ is the second join result:
    .join(t1, _.xmod == modf(t1))
    # _ is the third join result:
    .select(_.x, _.y, _.z)
    # Finally, _ is the selection result:
    .order_by(_.x)
)
```
