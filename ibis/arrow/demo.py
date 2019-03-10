"""
Demo for own Numpy API and comparison to Pandas API from Ibis
"""

import ibis
import pandas as pd
import pyarrow as pa


from arrow.api import connect


if __name__ == '__main__':
    # test Ibis framework for Pandas
    DF = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd'], 'baz': [1, 2, 1, 2],
                       'but': ['a', 'b', 'b', 'a'], 'buz': ['a', 'b', 'a', 'b']})
    DFS = {"t1": DF}
    CON = ibis.pandas.connect(DFS)
    TABLE = CON.table("t1")

    # test Arrow framework for Pandas
    DATA = [pa.array([1, 2, 3, 4]), pa.array(['a', 'b', 'c', 'd']), pa.array([1, 2, 1, 2]),
            pa.array(['a', 'b', 'b', 'a']), pa.array(['a', 'b', 'a', 'b'])]
    BATCH = pa.RecordBatch.from_arrays(DATA, ['foo', 'bar', 'baz', 'but', 'buz'])
    DFS = {"t1": BATCH}
    CON = connect(DFS)
    TABLE2 = CON.table("t1")

    # projections
    PROJ1 = TABLE.projection(["foo"])
    PROJ2 = TABLE2.projection(["bar"])

    print(PROJ1.execute())
    print(PROJ2.execute().to_pandas())

    # filtering
    FILTER1 = TABLE.filter([(TABLE.foo == 2) | (TABLE.bar == 'c')])
    FILTER2 = TABLE2.filter([(TABLE2.foo == 2) | (TABLE2.bar == 'c')])

    print(FILTER1.execute())
    print(FILTER2.execute().to_pandas())

    # group by and aggregation
    # sum
    SUM = TABLE.group_by([TABLE.baz]).aggregate(sum_of_foo=TABLE.foo.sum())
    SUM2 = TABLE2.group_by([TABLE2.baz]).aggregate(sum_of_foo=TABLE2.foo.sum())
    print(SUM.execute())
    print(SUM2.execute().to_pandas())

    # mean
    MEAN = TABLE.group_by([TABLE.baz]).aggregate(mean_of_foo=TABLE.foo.mean())
    MEAN2 = TABLE2.group_by([TABLE2.baz]).aggregate(mean_of_foo=TABLE2.foo.mean())
    print(MEAN.execute())
    print(MEAN2.execute().to_pandas())

    # count
    COUNT = TABLE.group_by([TABLE.baz]).aggregate(count_of_foo=TABLE.foo.count())
    COUNT2 = TABLE2.group_by([TABLE2.baz]).aggregate(count_of_foo=TABLE2.foo.count())
    print(COUNT.execute())
    print(COUNT2.execute().to_pandas())

    # min
    MIN = TABLE.group_by([TABLE.baz]).aggregate(min_of_foo=TABLE.foo.min())
    MIN2 = TABLE2.group_by([TABLE2.baz]).aggregate(min_of_foo=TABLE2.foo.min())
    print(MIN.execute())
    print(MIN2.execute().to_pandas())

    # max
    MAX = TABLE.group_by([TABLE.baz]).aggregate(max_of_foo=TABLE.foo.max())
    MAX2 = TABLE2.group_by([TABLE2.baz]).aggregate(max_of_foo=TABLE2.foo.max())
    print(MAX.execute())
    print(MAX2.execute().to_pandas())

    # chaining
    TEST = TABLE.group_by([TABLE.foo]).aggregate(min_of_baz=TABLE.baz.min())
    TEST = TEST.group_by([TEST.min_of_baz]).aggregate(sum_of_foo=TEST.foo.sum())
    print(TEST.execute())

    TEST2 = TABLE2.group_by([TABLE2.foo]).aggregate(min_of_baz=TABLE2.baz.min())
    TEST2 = TEST2.group_by([TEST2.min_of_baz]).aggregate(sum_of_foo=TEST2.foo.sum())
    print(TEST2.execute().to_pandas())
