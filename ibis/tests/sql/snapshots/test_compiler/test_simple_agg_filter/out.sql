SELECT t0.*
FROM (
  SELECT t1.*
  FROM my_table t1
  WHERE t1.`a` < 100
) t0
WHERE t0.`a` = (
  SELECT max(t1.`a`) AS `Max(a)`
  FROM my_table t1
  WHERE t1.`a` < 100
)