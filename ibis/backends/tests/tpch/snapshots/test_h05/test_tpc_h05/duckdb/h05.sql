SELECT
  t0.n_name,
  t0.revenue
FROM (
  SELECT
    t5.n_name AS n_name,
    SUM(t3.l_extendedprice * (
      CAST(1 AS TINYINT) - t3.l_discount
    )) AS revenue
  FROM main.customer AS t1
  JOIN main.orders AS t2
    ON t1.c_custkey = t2.o_custkey
  JOIN main.lineitem AS t3
    ON t3.l_orderkey = t2.o_orderkey
  JOIN main.supplier AS t4
    ON t3.l_suppkey = t4.s_suppkey
  JOIN main.nation AS t5
    ON t1.c_nationkey = t4.s_nationkey AND t4.s_nationkey = t5.n_nationkey
  JOIN main.region AS t6
    ON t5.n_regionkey = t6.r_regionkey
  WHERE
    t6.r_name = 'ASIA'
    AND t2.o_orderdate >= CAST('1994-01-01' AS DATE)
    AND t2.o_orderdate < CAST('1995-01-01' AS DATE)
  GROUP BY
    1
) AS t0
ORDER BY
  t0.revenue DESC