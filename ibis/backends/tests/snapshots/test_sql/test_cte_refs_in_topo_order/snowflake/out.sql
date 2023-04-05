WITH t0 AS (
  SELECT
    t4."key" AS "key"
  FROM "leaf" AS t4
  WHERE
    TRUE
), t1 AS (
  SELECT
    t0."key" AS "key"
  FROM t0
), t2 AS (
  SELECT
    t0."key" AS "key"
  FROM t0
  JOIN t1
    ON t0."key" = t1."key"
)
SELECT
  t2."key"
FROM t2
JOIN t2 AS t3
  ON t2."key" = t3."key"