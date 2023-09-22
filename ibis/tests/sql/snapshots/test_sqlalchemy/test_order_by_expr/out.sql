SELECT
  t0.a,
  t0.b
FROM (
  SELECT
    t1.a AS a,
    t1.b AS b
  FROM t AS t1
  WHERE
    t1.a = 1
) AS t0
ORDER BY
  CONCAT(t0.b, 'a') ASC NULLS FIRST