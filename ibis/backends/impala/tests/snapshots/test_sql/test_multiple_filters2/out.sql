WITH t0 AS (
  SELECT t1.*
  FROM `t0` t1
  WHERE t1.`a` < 100
)
SELECT t0.*
FROM t0
WHERE (t0.`a` = (
  SELECT max(t1.`a`) AS `Max(a)`
  FROM `t0` t1
  WHERE t1.`a` < 100
)) AND
      (t0.`b` = 'a')