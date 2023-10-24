WITH t0 AS (
  SELECT
    t2.l_suppkey AS l_suppkey,
    SUM(t2.l_extendedprice * (
      CAST(1 AS TINYINT) - t2.l_discount
    )) AS total_revenue
  FROM main.lineitem AS t2
  WHERE
    t2.l_shipdate >= CAST('1996-01-01' AS DATE)
    AND t2.l_shipdate < CAST('1996-04-01' AS DATE)
  GROUP BY
    1
)
SELECT
  t1.s_suppkey,
  t1.s_name,
  t1.s_address,
  t1.s_phone,
  t1.total_revenue
FROM (
  SELECT
    t2.s_suppkey AS s_suppkey,
    t2.s_name AS s_name,
    t2.s_address AS s_address,
    t2.s_nationkey AS s_nationkey,
    t2.s_phone AS s_phone,
    t2.s_acctbal AS s_acctbal,
    t2.s_comment AS s_comment,
    t0.l_suppkey AS l_suppkey,
    t0.total_revenue AS total_revenue
  FROM main.supplier AS t2
  JOIN t0
    ON t2.s_suppkey = t0.l_suppkey
  WHERE
    t0.total_revenue = (
      SELECT
        MAX(t0.total_revenue) AS "Max(total_revenue)"
      FROM t0
    )
) AS t1
ORDER BY
  t1.s_suppkey ASC