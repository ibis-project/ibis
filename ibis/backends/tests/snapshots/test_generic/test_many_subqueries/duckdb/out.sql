WITH t0 AS (
  SELECT
    t4.street AS street,
    ROW_NUMBER() OVER (ORDER BY t4.street ASC) - 1 AS key
  FROM data AS t4
), t1 AS (
  SELECT
    t0.key AS key
  FROM t0
), t2 AS (
  SELECT
    t0.street AS street,
    ROW_NUMBER() OVER (ORDER BY t0.street ASC) - 1 AS key
  FROM t0
  JOIN t1
    ON t0.key = t1.key
), t3 AS (
  SELECT
    t2.key AS key
  FROM t2
)
SELECT
  t2.street,
  t2.key
FROM t2
JOIN t3
  ON t2.key = t3.key