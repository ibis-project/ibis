WITH t0 AS (
  SELECT
    t4.*
  FROM `leaf` AS t4
  WHERE
    TRUE
), t1 AS (
  SELECT
    t0.`key`
  FROM t0
), t2 AS (
  SELECT
    t0.`key`
  FROM t0
  INNER JOIN t1
    ON t0.`key` = t1.`key`
)
SELECT
  t2.`key`
FROM t2
INNER JOIN t2 AS t3
  ON t2.`key` = t3.`key`