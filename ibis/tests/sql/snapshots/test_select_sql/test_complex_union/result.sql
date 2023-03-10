WITH t0 AS (
  SELECT t6.`diag`, t6.`status`
  FROM aids2_two t6
),
t1 AS (
  SELECT t6.`diag`, t6.`status`
  FROM aids2_one t6
),
t2 AS (
  SELECT t0.`diag` + 1 AS `diag`, t0.`status`
  FROM t0
),
t3 AS (
  SELECT t1.`diag` + 1 AS `diag`, t1.`status`
  FROM t1
),
t4 AS (
  SELECT CAST(t2.`diag` AS int) AS `diag`, t2.`status`
  FROM t2
),
t5 AS (
  SELECT CAST(t3.`diag` AS int) AS `diag`, t3.`status`
  FROM t3
)
SELECT *
FROM t5
UNION ALL
SELECT *
FROM t4