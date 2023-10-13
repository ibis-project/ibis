WITH t0 AS (
  SELECT
    t2.c_custkey AS c_custkey,
    t2.c_name AS c_name,
    t2.c_acctbal AS c_acctbal,
    t5.n_name AS n_name,
    t2.c_address AS c_address,
    t2.c_phone AS c_phone,
    t2.c_comment AS c_comment,
    SUM(t4.l_extendedprice * (
      1 - t4.l_discount
    )) AS revenue
  FROM "hive".ibis_sf1.customer AS t2
  JOIN "hive".ibis_sf1.orders AS t3
    ON t2.c_custkey = t3.o_custkey
  JOIN "hive".ibis_sf1.lineitem AS t4
    ON t4.l_orderkey = t3.o_orderkey
  JOIN "hive".ibis_sf1.nation AS t5
    ON t2.c_nationkey = t5.n_nationkey
  WHERE
    t3.o_orderdate >= FROM_ISO8601_DATE('1993-10-01')
    AND t3.o_orderdate < FROM_ISO8601_DATE('1994-01-01')
    AND t4.l_returnflag = 'R'
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
)
SELECT
  t1.c_custkey,
  t1.c_name,
  t1.revenue,
  t1.c_acctbal,
  t1.n_name,
  t1.c_address,
  t1.c_phone,
  t1.c_comment
FROM (
  SELECT
    t0.c_custkey AS c_custkey,
    t0.c_name AS c_name,
    t0.revenue AS revenue,
    t0.c_acctbal AS c_acctbal,
    t0.n_name AS n_name,
    t0.c_address AS c_address,
    t0.c_phone AS c_phone,
    t0.c_comment AS c_comment
  FROM t0
) AS t1
ORDER BY
  t1.revenue DESC
LIMIT 20