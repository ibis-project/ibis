WITH t0 AS (
  SELECT t5.`diag`, t5.`status`
  FROM aids2_two t5
),
t1 AS (
  SELECT t5.`diag`, t5.`status`
  FROM aids2_one t5
),
t2 AS (
  SELECT t0.`diag` + 1 AS `diag`, t0.`status`
  FROM t0
),
t3 AS (
  SELECT t1.`diag` + 1 AS `diag`, t1.`status`
  FROM t1
)
SELECT t4.`diag`, t4.`status`
FROM (
  WITH t0 AS (
    SELECT t5.`diag`, t5.`status`
    FROM aids2_two t5
  ),
  t1 AS (
    SELECT t5.`diag`, t5.`status`
    FROM aids2_one t5
  ),
  t2 AS (
    SELECT t0.`diag` + 1 AS `diag`, t0.`status`
    FROM t0
  ),
  t3 AS (
    SELECT t1.`diag` + 1 AS `diag`, t1.`status`
    FROM t1
  ),
  t5 AS (
    SELECT CAST(t2.`diag` AS int) AS `diag`, t2.`status`
    FROM t2
  ),
  t6 AS (
    SELECT CAST(t3.`diag` AS int) AS `diag`, t3.`status`
    FROM t3
  )
  SELECT *
  FROM t6
  UNION ALL
  SELECT *
  FROM t5
) t4