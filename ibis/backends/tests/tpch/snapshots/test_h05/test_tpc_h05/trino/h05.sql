SELECT
  t0.n_name,
  t0.revenue
FROM (
  SELECT
    t5.n_name AS n_name,
    SUM(t3.l_extendedprice * (
      1 - t3.l_discount
    )) AS revenue
  FROM hive.ibis_sf1.customer AS t1
  JOIN hive.ibis_sf1.orders AS t2
    ON t1.c_custkey = t2.o_custkey
  JOIN hive.ibis_sf1.lineitem AS t3
    ON t3.l_orderkey = t2.o_orderkey
  JOIN hive.ibis_sf1.supplier AS t4
    ON t3.l_suppkey = t4.s_suppkey
  JOIN hive.ibis_sf1.nation AS t5
    ON t1.c_nationkey = t4.s_nationkey AND t4.s_nationkey = t5.n_nationkey
  JOIN hive.ibis_sf1.region AS t6
    ON t5.n_regionkey = t6.r_regionkey
  WHERE
    t6.r_name = 'ASIA'
    AND t2.o_orderdate >= FROM_ISO8601_DATE('1994-01-01')
    AND t2.o_orderdate < FROM_ISO8601_DATE('1995-01-01')
  GROUP BY
    1
) AS t0
ORDER BY
  t0.revenue DESC