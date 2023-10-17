SELECT
  t0.`a`
FROM (
  SELECT
    t1.*
  FROM t0 AS t1
  UNION DISTINCT
  SELECT
    t1.*
  FROM t1
) AS t0