WITH t0 AS (
  SELECT
    t6.n_name AS supp_nation,
    t7.n_name AS cust_nation,
    t3.l_shipdate AS l_shipdate,
    t3.l_extendedprice AS l_extendedprice,
    t3.l_discount AS l_discount,
    CAST(EXTRACT(year FROM t3.l_shipdate) AS SMALLINT) AS l_year,
    t3.l_extendedprice * (
      1 - t3.l_discount
    ) AS volume
  FROM hive.ibis_sf1.supplier AS t2
  JOIN hive.ibis_sf1.lineitem AS t3
    ON t2.s_suppkey = t3.l_suppkey
  JOIN hive.ibis_sf1.orders AS t4
    ON t4.o_orderkey = t3.l_orderkey
  JOIN hive.ibis_sf1.customer AS t5
    ON t5.c_custkey = t4.o_custkey
  JOIN hive.ibis_sf1.nation AS t6
    ON t2.s_nationkey = t6.n_nationkey
  JOIN hive.ibis_sf1.nation AS t7
    ON t5.c_nationkey = t7.n_nationkey
)
SELECT
  t1.supp_nation,
  t1.cust_nation,
  t1.l_year,
  t1.revenue
FROM (
  SELECT
    t0.supp_nation AS supp_nation,
    t0.cust_nation AS cust_nation,
    t0.l_year AS l_year,
    SUM(t0.volume) AS revenue
  FROM t0
  WHERE
    (
      t0.cust_nation = 'FRANCE' AND t0.supp_nation = 'GERMANY'
      OR t0.cust_nation = 'GERMANY'
      AND t0.supp_nation = 'FRANCE'
    )
    AND t0.l_shipdate BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
  GROUP BY
    1,
    2,
    3
) AS t1
ORDER BY
  t1.supp_nation ASC,
  t1.cust_nation ASC,
  t1.l_year ASC