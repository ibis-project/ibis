.. _self_joins:

**********
Self joins
**********

If you’re a relational data guru, you may have wondered how it’s
possible to join tables with themselves, because joins clauses involve
column references back to the original table.

Consider the SQL

.. code:: sql

       SELECT t1.key, sum(t1.value - t2.value) AS metric
       FROM my_table t1
         JOIN my_table t2
           ON t1.key = t2.subkey
       GROUP BY 1

Here, we have an unambiguous way to refer to each of the tables through
aliasing.

Let’s consider the TPC-H database, and support we want to compute
year-over-year change in total order amounts by region using joins.

.. code:: python

    >>> region = con.table('tpch_region')
    >>> nation = con.table('tpch_nation')
    >>> customer = con.table('tpch_customer')
    >>> orders = con.table('tpch_orders')
    >>> orders.limit(5)
       o_orderkey  o_custkey o_orderstatus o_totalprice o_orderdate  \
    0           1      36901             O    173665.47  1996-01-02
    1           2      78002             O     46929.18  1996-12-01
    2           3     123314             F    193846.25  1993-10-14
    3           4     136777             O     32151.78  1995-10-11
    4           5      44485             F    144659.20  1994-07-30

      o_orderpriority          o_clerk  o_shippriority  \
    0           5-LOW  Clerk#000000951               0
    1        1-URGENT  Clerk#000000880               0
    2           5-LOW  Clerk#000000955               0
    3           5-LOW  Clerk#000000124               0
    4           5-LOW  Clerk#000000925               0

                                               o_comment
    0                 nstructions sleep furiously among
    1   foxes. pending accounts at the pending, silen...
    2  sly final accounts boost. carefully regular id...
    3  sits. slyly regular warthogs cajole. regular, ...
    4  quickly. bold deposits sleep slyly. packages u...

First, let’s join all the things and select the fields we care about:

.. code:: python

    >>> fields_of_interest = [region.r_name.name('region'),
    ...                       nation.n_name.name('nation'),
    ...                       orders.o_totalprice.name('amount'),
    ...                       orders.o_orderdate.cast('timestamp').name('odate') # these are strings
    ...                       ]
    >>> joined_all = (region.join(nation, region.r_regionkey == nation.n_regionkey)
    ...               .join(customer, customer.c_nationkey == nation.n_nationkey)
    ...               .join(orders, orders.o_custkey == customer.c_custkey)
    ...               [fields_of_interest])

Okay, great, let’s have a look:

.. code:: python

    >>> joined_all.limit(5)
            region         nation     amount      odate
    0      AMERICA  UNITED STATES  160843.35 1992-06-22
    1  MIDDLE EAST           IRAN   78307.91 1996-04-19
    2       EUROPE         FRANCE  103237.90 1994-10-12
    3       EUROPE         FRANCE  201463.59 1997-09-12
    4         ASIA          JAPAN  166098.86 1995-09-12

Sweet, now let’s aggregate by year and region:

.. code:: python

    >>> year = joined_all.odate.year().name('year')
    >>> total = joined_all.amount.sum().cast('double').name('total')
    >>> annual_amounts = (joined_all
    ...                   .group_by(['region', year])
    ...                   .aggregate(total))
        >>> annual_amounts.limit(5)
             region  year         total
    0        EUROPE  1994  6.979473e+09
    1        EUROPE  1996  7.015421e+09
    2          ASIA  1997  6.910663e+09
    3          ASIA  1998  4.058824e+09
    4        EUROPE  1992  6.926705e+09

Looking good so far. Now, we need to join this table on itself, by
subtracting 1 from one of the year columns.

We do this by creating a “joinable” view of a table that is considered a
distinct object within Ibis. To do this, use the ``view`` function:

.. code:: python

    >>> current = annual_amounts
    >>> prior = annual_amounts.view()
    >>> yoy_change = (current.total - prior.total).name('yoy_change')
    >>> results = (current.join(prior, ((current.region == prior.region) &
    ...                                 (current.year == (prior.year - 1))))
    ...            [current.region, current.year, yoy_change])
    >>> df = results.execute()

.. code:: python

    >>> df['yoy_pretty'] = df.yoy_change.map(lambda x: '$%.2fmm' % (x / 1000000.))

If you’re being fastidious and want to consider the first year occurring
in the dataset for each region to have 0 for the prior year, you will
instead need to do an outer join and treat nulls in the prior side of
the join as zero:

.. code:: python

    >>> yoy_change = (current.total - prior.total.zeroifnull()).name('yoy_change')
    >>> results = (current.outer_join(prior, ((current.region == prior.region) &
    ...                                       (current.year == (prior.year - 1))))
    ...            [current.region, current.year, current.total,
    ...             prior.total.zeroifnull().name('prior_total'),
    ...             yoy_change])
    >>> results.limit(5)
            region  year         total   prior_total    yoy_change
    0         ASIA  1998  4.058824e+09  0.000000e+00  4.058824e+09
    1       AFRICA  1994  6.837587e+09  6.908429e+09 -7.084172e+07
    2      AMERICA  1996  6.883057e+09  6.922465e+09 -3.940791e+07
    3       AFRICA  1996  6.878112e+09  6.848983e+09  2.912979e+07
    4       AFRICA  1992  6.873319e+09  6.859733e+09  1.358699e+07
