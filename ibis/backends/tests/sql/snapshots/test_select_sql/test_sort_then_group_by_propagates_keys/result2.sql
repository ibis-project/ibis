SELECT
  t2.b AS b,
  COUNT(*) AS b_count
FROM (
  SELECT
    t1.b AS b
  FROM (
    SELECT
      *
    FROM t AS t0
    ORDER BY
      t0.b ASC
  ) AS t1
) AS t2
GROUP BY
  1