WITH t0 AS (
  SELECT
    t2.l_orderkey AS l_orderkey,
    SUM(t2.l_quantity) AS qty_sum
  FROM hive.ibis_sf1.lineitem AS t2
  GROUP BY
    1
)
SELECT
  t1.c_name,
  t1.c_custkey,
  t1.o_orderkey,
  t1.o_orderdate,
  t1.o_totalprice,
  t1.sum_qty
FROM (
  SELECT
    t2.c_name AS c_name,
    t2.c_custkey AS c_custkey,
    t3.o_orderkey AS o_orderkey,
    t3.o_orderdate AS o_orderdate,
    t3.o_totalprice AS o_totalprice,
    SUM(t4.l_quantity) AS sum_qty
  FROM hive.ibis_sf1.customer AS t2
  JOIN hive.ibis_sf1.orders AS t3
    ON t2.c_custkey = t3.o_custkey
  JOIN hive.ibis_sf1.lineitem AS t4
    ON t3.o_orderkey = t4.l_orderkey
  WHERE
    t3.o_orderkey IN (
      SELECT
        t5.l_orderkey
      FROM (
        SELECT
          t0.l_orderkey AS l_orderkey,
          t0.qty_sum AS qty_sum
        FROM t0
        WHERE
          t0.qty_sum > 300
      ) AS t5
    )
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS t1
ORDER BY
  t1.o_totalprice DESC,
  t1.o_orderdate ASC
LIMIT 100