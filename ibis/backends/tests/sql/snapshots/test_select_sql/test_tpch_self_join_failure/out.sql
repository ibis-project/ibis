SELECT
  t8.region AS region,
  t8.year AS year,
  t8.total - t9.total AS yoy_change
FROM (
  SELECT
    t7.region AS region,
    EXTRACT('year' FROM t7.odate) AS year,
    CAST(SUM(t7.amount) AS DOUBLE) AS total
  FROM (
    SELECT
      t0.r_name AS region,
      t1.n_name AS nation,
      t3.o_totalprice AS amount,
      CAST(t3.o_orderdate AS TIMESTAMP) AS odate
    FROM tpch_region AS t0
    INNER JOIN tpch_nation AS t1
      ON t0.r_regionkey = t1.n_regionkey
    INNER JOIN tpch_customer AS t2
      ON t2.c_nationkey = t1.n_nationkey
    INNER JOIN tpch_orders AS t3
      ON t3.o_custkey = t2.c_custkey
  ) AS t7
  GROUP BY
    1,
    2
) AS t8
INNER JOIN (
  SELECT
    t7.region AS region,
    EXTRACT('year' FROM t7.odate) AS year,
    CAST(SUM(t7.amount) AS DOUBLE) AS total
  FROM (
    SELECT
      t0.r_name AS region,
      t1.n_name AS nation,
      t3.o_totalprice AS amount,
      CAST(t3.o_orderdate AS TIMESTAMP) AS odate
    FROM tpch_region AS t0
    INNER JOIN tpch_nation AS t1
      ON t0.r_regionkey = t1.n_regionkey
    INNER JOIN tpch_customer AS t2
      ON t2.c_nationkey = t1.n_nationkey
    INNER JOIN tpch_orders AS t3
      ON t3.o_custkey = t2.c_custkey
  ) AS t7
  GROUP BY
    1,
    2
) AS t9
  ON t8.year = (
    t9.year - CAST(1 AS TINYINT)
  )