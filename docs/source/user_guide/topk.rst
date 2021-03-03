.. _topk:

*****************
“Top-K” Filtering
*****************

A common analytical pattern involves subsetting based on some method of
ranking. For example, “the 5 most frequently occurring widgets in a
dataset”. By choosing the right metric, you can obtain the most
important or least important items from some dimension, for some
definition of important.

To carry out the pattern by hand involves the following

-  Choose a ranking metric
-  Aggregate, computing the ranking metric, by the target dimension
-  Order by the ranking metric and take the highest K values
-  Use those values as a set filter (either with ``semi_join`` or
   ``isin``) in your next query

For example, let’s look at the TPC-H tables and find the 5 or 10
customers who placed the most orders over their lifetime:

.. code:: python

    >>> orders = con.table('tpch_orders')
    >>> top_orders = (orders
    ...               .group_by('o_custkey')
    ...               .size()
    ...               .sort_by(('count', False))
    ...               .limit(5))
    >>> top_orders
       o_custkey  count
    0       3451     41
    1     102022     41
    2     102004     41
    3      79300     40
    4     117082     40

Now, we could use these customer keys as a filter in some other
analysis:

.. code:: python

    >>> # Among the top 5 most frequent customers, what's the histogram of their order statuses?
    >>> analysis = (orders[orders.o_custkey.isin(top_orders.o_custkey)]
    ...             .group_by('o_orderstatus')
    ...             .size())
    >>> analysis
      o_orderstatus  count
    0             P      5
    1             F     85
    2             O    113

This is such a common pattern that Ibis supports a high level primitive
``topk`` operation, which can be used immediately as a filter:

.. code:: python

    >>> top_orders = orders.o_custkey.topk(5)
    >>> orders[top_orders].group_by('o_orderstatus').size()
      o_orderstatus  count
    0             P      5
    1             F     85
    2             O    113

This goes a little further. Suppose now we want to rank customers by
their total spending instead of the number of orders, perhaps a more
meaningful metric:

.. code:: python

    >>> total_spend = orders.o_totalprice.sum().name('total')
    >>> top_spenders = (orders
    ...                .group_by('o_custkey')
    ...                .aggregate(total_spend)
    ...                .sort_by(('total', False))
    ...                .limit(5))
    >>> top_spenders
       o_custkey       total
    0     143500  7012696.48
    1      95257  6563511.23
    2      87115  6457526.26
    3     131113  6311428.86
    4     103834  6306524.23

To use another metric, just pass it to the ``by`` argument in ``topk``:

.. code:: python

    >>> top_spenders = orders.o_custkey.topk(5, by=total_spend)
    >>> orders[top_spenders].group_by('o_orderstatus').size()
      o_orderstatus  count
    0             P      1
    1             F     78
    2             O     98
