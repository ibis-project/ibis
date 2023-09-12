WITH t0 AS (
  SELECT
    t3.foo_id AS foo_id,
    SUM(t3.f) AS total
  FROM star1 AS t3
  GROUP BY
    1
), t1 AS (
  SELECT
    t0.foo_id AS foo_id,
    t0.total AS total,
    t3.value1 AS value1
  FROM t0
  JOIN star2 AS t3
    ON t0.foo_id = t3.foo_id
)
SELECT
  t2.foo_id,
  t2.total,
  t2.value1
FROM (
  SELECT
    t1.foo_id AS foo_id,
    t1.total AS total,
    t1.value1 AS value1
  FROM t1
  WHERE
    t1.total > 100
) AS t2
ORDER BY
  t2.total DESC