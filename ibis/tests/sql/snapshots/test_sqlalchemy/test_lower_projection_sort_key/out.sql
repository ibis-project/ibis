WITH t1 AS (
  SELECT
    t3.foo_id AS foo_id,
    SUM(t3.f) AS total
  FROM star1 AS t3
  GROUP BY
    1
), t3 AS (
  SELECT
    t1.foo_id AS foo_id,
    t1.total AS total,
    t4.value1 AS value1
  FROM t1
  JOIN star2 AS t4
    ON t1.foo_id = t4.foo_id
)
SELECT
  t0.foo_id,
  t0.total,
  t0.value1
FROM (
  SELECT
    t3.foo_id AS foo_id,
    t3.total AS total,
    t3.value1 AS value1
  FROM t3
  WHERE
    t3.total > 100
) AS t0
ORDER BY
  t0.total DESC