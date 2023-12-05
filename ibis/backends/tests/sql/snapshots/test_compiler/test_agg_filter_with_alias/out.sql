WITH t0 AS (
  SELECT t1.*, t1.`b` * 2 AS `b2`
  FROM my_table t1
)
SELECT t0.`a`, t0.`b2`
FROM t0
WHERE (t0.`a` < 100) AND
      (t0.`a` = (
  SELECT max(t0.`a`) AS `Max(a)`
  FROM t0
))