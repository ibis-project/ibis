# Compute the top K records

<!-- prettier-ignore-start -->
Here we use the [`topk`][ibis.expr.types.Column.topk] method to compute the top
5 customers for some generated TPC-H data by:
<!-- prettier-ignore-end -->

- count (the default)
- sum of order totals

```python
>>> import ibis
>>> ibis.options.interactive = True
>>> con = ibis.duckdb.connect()  # in-memory duckdb
>>> con.raw_sql("CALL dbgen(sf=0.1)")
>>> orders = con.table("orders")
>>> orders.o_custkey.topk(5)  # top 5 most frequent customers
┏━━━━━━━━━━━┳━━━━━━━┓
┃ o_custkey ┃ count ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ !int32    │ int64 │
├───────────┼───────┤
│      8761 │    36 │
│     11998 │    36 │
│      3151 │    35 │
│      8362 │    35 │
│       388 │    35 │
└───────────┴───────┘
>>> topk = orders.o_custkey.topk(5, by=orders.o_totalprice.sum())  # top 5 largest spending customers
>>> topk
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ o_custkey ┃ sum            ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ !int32    │ decimal(38, 2) │
├───────────┼────────────────┤
│      8362 │     5793605.05 │
│      6958 │     5370682.19 │
│      9454 │     5354381.81 │
│       346 │     5323350.43 │
│     10354 │     5227957.24 │
└───────────┴────────────────┘
```

<!-- prettier-ignore-start -->
You can also use [`topk`][ibis.expr.types.Column.topk] to retrieve the rows
from the original table that match the key used, in this case `o_custkey`. This
is done with a left semi join:
<!-- prettier-ignore-end -->

```python
>>> orders.semi_join(topk, "o_custkey")
┏━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ o_orderkey ┃ o_custkey ┃ o_orderstatus ┃ o_totalprice    ┃ o_orderdate ┃ o_orderpriority ┃ o_clerk         ┃ o_shippriority ┃ o_comment                                                    ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ !int32     │ !int32    │ !string       │ !decimal(15, 2) │ !date       │ !string         │ !string         │ !int32         │ !string                                                      │
├────────────┼───────────┼───────────────┼─────────────────┼─────────────┼─────────────────┼─────────────────┼────────────────┼──────────────────────────────────────────────────────────────┤
│       4000 │      6958 │ F             │       115722.85 │ 1992-01-04  │ 5-LOW           │ Clerk#000000339 │              0 │ le carefully closely even pinto beans. regular, ironic foxe… │
│      14402 │      8362 │ F             │       131557.79 │ 1993-10-15  │ 3-MEDIUM        │ Clerk#000000672 │              0 │ azzle slyly. carefully regular instructions affix carefully… │
│      14784 │     10354 │ F             │       216307.34 │ 1992-03-15  │ 3-MEDIUM        │ Clerk#000000479 │              0 │ lyly final theodoli                                          │
│      17415 │     10354 │ O             │       110427.40 │ 1996-09-18  │ 2-HIGH          │ Clerk#000000148 │              0 │ . furiously even asymptotes wake carefully according to t    │
│      17760 │      9454 │ F             │       167249.60 │ 1992-06-05  │ 4-NOT SPECIFIED │ Clerk#000000093 │              0 │ uriously final pinto beans wake furiously                    │
│      18853 │      9454 │ F             │       163677.19 │ 1993-01-18  │ 1-URGENT        │ Clerk#000000046 │              0 │ sts. courts haggle furiously. even, enticing depo            │
│      21317 │      8362 │ P             │       267386.98 │ 1995-04-10  │ 5-LOW           │ Clerk#000000737 │              0 │ Tiresias. accounts a                                         │
│      23138 │      8362 │ O             │       174882.01 │ 1997-07-23  │ 1-URGENT        │ Clerk#000000253 │              0 │ uctions integrate carefully regular pinto beans. silent acc… │
│      23972 │     10354 │ F             │       129646.66 │ 1993-08-17  │ 4-NOT SPECIFIED │ Clerk#000001000 │              0 │ s. blithely final packages sleep quickly idle pearls. even,… │
│      24064 │       346 │ F             │       147095.22 │ 1993-07-26  │ 3-MEDIUM        │ Clerk#000000020 │              0 │ ithely final foxes. furiously final instructi                │
│          … │         … │ …             │               … │ …           │ …               │ …               │              … │ …                                                            │
└────────────┴───────────┴───────────────┴─────────────────┴─────────────┴─────────────────┴─────────────────┴────────────────┴──────────────────────────────────────────────────────────────┘
```
