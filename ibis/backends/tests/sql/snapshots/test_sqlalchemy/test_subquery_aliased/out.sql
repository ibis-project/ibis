WITH t0 AS (
  SELECT
    t2.foo_id AS foo_id,
    SUM(t2.f) AS total
  FROM star1 AS t2
  GROUP BY
    1
)
SELECT
  t0.foo_id,
  t0.total,
  t1.value1
FROM t0
JOIN star2 AS t1
  ON t0.foo_id = t1.foo_id