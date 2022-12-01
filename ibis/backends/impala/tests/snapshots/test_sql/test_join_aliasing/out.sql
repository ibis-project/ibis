WITH t0 AS (
  SELECT `d`, `c`
  FROM t2
),
t1 AS (
  SELECT `d`, CAST(`d` / 15 AS bigint) AS `idx`, `c`, count(1) AS `row_count`
  FROM t0
  GROUP BY 1, 2, 3
),
t2 AS (
  SELECT *, `a` + 20 AS `d`
  FROM test_table
)
SELECT t3.*, t4.`total`
FROM (
  SELECT `d`, `b`, count(1) AS `count`, count(DISTINCT `c`) AS `unique`
  FROM t2
  GROUP BY 1, 2
) t3
  INNER JOIN (
    SELECT t5.*
    FROM (
      SELECT t1.*, t8.`total`
      FROM t1
        INNER JOIN (
          SELECT `d`, sum(`row_count`) AS `total`
          FROM t1
          GROUP BY 1
        ) t8
          ON t1.`d` = t8.`d`
    ) t5
    WHERE t5.`row_count` < (t5.`total` / 2)
  ) t4
    ON t3.`d` = t4.`d`