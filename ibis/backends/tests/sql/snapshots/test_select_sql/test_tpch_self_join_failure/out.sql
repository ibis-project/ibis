SELECT
  t13.region,
  t13.year,
  t13.total - t15.total AS yoy_change
FROM (
  SELECT
    t11.region,
    EXTRACT('year' FROM t11.odate) AS year,
    CAST(SUM(t11.amount) AS DOUBLE) AS total
  FROM (
    SELECT
      t4.r_name AS region,
      t5.n_name AS nation,
      t7.o_totalprice AS amount,
      CAST(t7.o_orderdate AS TIMESTAMP) AS odate
    FROM tpch_region AS t4
    INNER JOIN tpch_nation AS t5
      ON t4.r_regionkey = t5.n_regionkey
    INNER JOIN tpch_customer AS t6
      ON t6.c_nationkey = t5.n_nationkey
    INNER JOIN tpch_orders AS t7
      ON t7.o_custkey = t6.c_custkey
  ) AS t11
  GROUP BY
    1,
    2
) AS t13
INNER JOIN (
  SELECT
    t11.region,
    EXTRACT('year' FROM t11.odate) AS year,
    CAST(SUM(t11.amount) AS DOUBLE) AS total
  FROM (
    SELECT
      t4.r_name AS region,
      t5.n_name AS nation,
      t7.o_totalprice AS amount,
      CAST(t7.o_orderdate AS TIMESTAMP) AS odate
    FROM tpch_region AS t4
    INNER JOIN tpch_nation AS t5
      ON t4.r_regionkey = t5.n_regionkey
    INNER JOIN tpch_customer AS t6
      ON t6.c_nationkey = t5.n_nationkey
    INNER JOIN tpch_orders AS t7
      ON t7.o_custkey = t6.c_custkey
  ) AS t11
  GROUP BY
    1,
    2
) AS t15
  ON t13.year = (
    t15.year - CAST(1 AS TINYINT)
  )