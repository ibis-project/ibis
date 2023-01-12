WITH t0 AS (
  SELECT t2.`d`, t2.`c`
  FROM t2
),
t1 AS (
  SELECT t0.`d`, CAST(t0.`d` / 15 AS bigint) AS `idx`, t0.`c`,
         count(1) AS `row_count`
  FROM t0
  GROUP BY 1, 2, 3
),
t2 AS (
  SELECT t5.*, t5.`a` + 20 AS `d`
  FROM test_table t5
)
SELECT t3.*, t4.`total`
FROM (
  SELECT t2.`d`, t2.`b`, count(1) AS `count`,
         count(DISTINCT t2.`c`) AS `unique`
  FROM t2
  GROUP BY 1, 2
) t3
  INNER JOIN (
    SELECT t5.*
    FROM (
      SELECT t1.*, t7.`total`
      FROM t1
        INNER JOIN (
          SELECT t1.`d`, sum(t1.`row_count`) AS `total`
          FROM t1
          GROUP BY 1
        ) t7
          ON t1.`d` = t7.`d`
    ) t5
    WHERE t5.`row_count` < (t5.`total` / 2)
  ) t4
    ON t3.`d` = t4.`d`