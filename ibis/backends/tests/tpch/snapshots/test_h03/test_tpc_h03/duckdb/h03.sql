WITH t0 AS (
  SELECT
    t4.l_orderkey AS l_orderkey,
    t3.o_orderdate AS o_orderdate,
    t3.o_shippriority AS o_shippriority,
    SUM(t4.l_extendedprice * (
      CAST(1 AS TINYINT) - t4.l_discount
    )) AS revenue
  FROM main.customer AS t2
  JOIN main.orders AS t3
    ON t2.c_custkey = t3.o_custkey
  JOIN main.lineitem AS t4
    ON t4.l_orderkey = t3.o_orderkey
  WHERE
    t2.c_mktsegment = 'BUILDING'
    AND t3.o_orderdate < CAST('1995-03-15' AS DATE)
    AND t4.l_shipdate > CAST('1995-03-15' AS DATE)
  GROUP BY
    1,
    2,
    3
)
SELECT
  t1.l_orderkey,
  t1.revenue,
  t1.o_orderdate,
  t1.o_shippriority
FROM (
  SELECT
    t0.l_orderkey AS l_orderkey,
    t0.revenue AS revenue,
    t0.o_orderdate AS o_orderdate,
    t0.o_shippriority AS o_shippriority
  FROM t0
) AS t1
ORDER BY
  t1.revenue DESC,
  t1.o_orderdate ASC
LIMIT 10