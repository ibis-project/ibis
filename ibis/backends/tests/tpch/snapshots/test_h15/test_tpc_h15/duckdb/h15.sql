WITH t0 AS (
  SELECT
    t3.l_suppkey AS l_suppkey,
    SUM(t3.l_extendedprice * (
      CAST(1 AS TINYINT) - t3.l_discount
    )) AS total_revenue
  FROM main.lineitem AS t3
  WHERE
    t3.l_shipdate >= CAST('1996-01-01' AS DATE)
    AND t3.l_shipdate < CAST('1996-04-01' AS DATE)
  GROUP BY
    1
), t1 AS (
  SELECT
    t3.s_suppkey AS s_suppkey,
    t3.s_name AS s_name,
    t3.s_address AS s_address,
    t3.s_nationkey AS s_nationkey,
    t3.s_phone AS s_phone,
    t3.s_acctbal AS s_acctbal,
    t3.s_comment AS s_comment,
    t0.l_suppkey AS l_suppkey,
    t0.total_revenue AS total_revenue
  FROM main.supplier AS t3
  JOIN t0
    ON t3.s_suppkey = t0.l_suppkey
  WHERE
    t0.total_revenue = (
      SELECT
        MAX(t0.total_revenue) AS "Max(total_revenue)"
      FROM t0
    )
)
SELECT
  t2.s_suppkey,
  t2.s_name,
  t2.s_address,
  t2.s_phone,
  t2.total_revenue
FROM (
  SELECT
    t1.s_suppkey AS s_suppkey,
    t1.s_name AS s_name,
    t1.s_address AS s_address,
    t1.s_nationkey AS s_nationkey,
    t1.s_phone AS s_phone,
    t1.s_acctbal AS s_acctbal,
    t1.s_comment AS s_comment,
    t1.l_suppkey AS l_suppkey,
    t1.total_revenue AS total_revenue
  FROM t1
  ORDER BY
    t1.s_suppkey ASC
) AS t2