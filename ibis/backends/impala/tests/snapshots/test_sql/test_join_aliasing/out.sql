WITH t0 AS (
  SELECT t7.*, t7.`a` + 20 AS `d`
  FROM `test_table` t7
),
t1 AS (
  SELECT t0.`d`, t0.`c`
  FROM t0
),
t2 AS (
  SELECT t1.`d`, CAST(t1.`d` / 15 AS bigint) AS `idx`, t1.`c`,
         count(1) AS `row_count`
  FROM t1
  GROUP BY 1, 2, 3
),
t3 AS (
  SELECT t2.`d`, sum(t2.`row_count`) AS `total`
  FROM t2
  GROUP BY 1
),
t4 AS (
  SELECT t2.*, t3.`total`
  FROM t2
    INNER JOIN t3
      ON t2.`d` = t3.`d`
),
t5 AS (
  SELECT t4.*
  FROM t4
  WHERE t4.`row_count` < (t4.`total` / 2)
)
SELECT t6.*, t5.`total`
FROM (
  SELECT t0.`d`, t0.`b`, count(1) AS `count`,
         count(DISTINCT t0.`c`) AS `unique`
  FROM t0
  GROUP BY 1, 2
) t6
  INNER JOIN t5
    ON t6.`d` = t5.`d`