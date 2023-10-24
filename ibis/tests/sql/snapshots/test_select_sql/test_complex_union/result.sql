WITH t0 AS (
  SELECT t3.`diag` + 1 AS `diag`, t3.`status`
  FROM aids2_two t3
),
t1 AS (
  SELECT t3.`diag` + 1 AS `diag`, t3.`status`
  FROM aids2_one t3
)
SELECT t2.`diag`, t2.`status`
FROM (
  WITH t0 AS (
    SELECT t3.`diag` + 1 AS `diag`, t3.`status`
    FROM aids2_two t3
  ),
  t1 AS (
    SELECT t3.`diag` + 1 AS `diag`, t3.`status`
    FROM aids2_one t3
  ),
  t3 AS (
    SELECT CAST(t0.`diag` AS int) AS `diag`, t0.`status`
    FROM t0
  ),
  t4 AS (
    SELECT CAST(t1.`diag` AS int) AS `diag`, t1.`status`
    FROM t1
  )
  SELECT *
  FROM t4
  UNION ALL
  SELECT *
  FROM t3
) t2