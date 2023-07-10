SELECT
  t0.foo AS foo,
  t0.bar AS bar,
  t0.city AS city,
  t0.v1 AS v1,
  t0.v2 AS v2
FROM tbl AS t0
SEMI JOIN (
  SELECT
    *
  FROM (
    SELECT
      t0.city AS city,
      COUNT(t0.city) AS "Count(city)"
    FROM tbl AS t0
    GROUP BY
      1
  ) AS t1
  ORDER BY
    t1."Count(city)" DESC
  LIMIT 10
) AS t3
  ON t0.city = t3.city