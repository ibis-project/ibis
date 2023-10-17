WITH t0 AS (
  SELECT
    t4.l_orderkey AS l_orderkey,
    t3.o_orderdate AS o_orderdate,
    t3.o_shippriority AS o_shippriority,
    SUM(t4.l_extendedprice * (
      1 - t4.l_discount
    )) AS revenue
  FROM hive.ibis_sf1.customer AS t2
  JOIN hive.ibis_sf1.orders AS t3
    ON t2.c_custkey = t3.o_custkey
  JOIN hive.ibis_sf1.lineitem AS t4
    ON t4.l_orderkey = t3.o_orderkey
  WHERE
    t2.c_mktsegment = 'BUILDING'
    AND t3.o_orderdate < FROM_ISO8601_DATE('1995-03-15')
    AND t4.l_shipdate > FROM_ISO8601_DATE('1995-03-15')
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