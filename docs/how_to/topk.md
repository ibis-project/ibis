# Compute the Top K Records

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
   o_custkey  count
0      11998     36
1       8761     36
2       3151     35
3        388     35
4       8362     35

>>> orders.o_custkey.topk(5, by=orders.o_totalprice.sum())  # top 5 largest spending customers
   o_custkey         sum
0       8362  5793605.05
1       6958  5370682.19
2       9454  5354381.81
3        346  5323350.43
4      10354  5227957.24
```
