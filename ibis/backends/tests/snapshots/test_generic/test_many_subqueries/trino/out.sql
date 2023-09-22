WITH t0 AS (
  SELECT
    t5.street AS street,
    ROW_NUMBER() OVER (ORDER BY t5.street ASC) - 1 AS key
  FROM data AS t5
), t1 AS (
  SELECT
    t0.key AS key
  FROM t0
), t2 AS (
  SELECT
    t0.street AS street,
    t0.key AS key
  FROM t0
  JOIN t1
    ON t0.key = t1.key
), t3 AS (
  SELECT
    t2.street AS street,
    ROW_NUMBER() OVER (ORDER BY t2.street ASC) - 1 AS key
  FROM t2
), t4 AS (
  SELECT
    t3.key AS key
  FROM t3
)
SELECT
  t3.street,
  t3.key
FROM t3
JOIN t4
  ON t3.key = t4.key