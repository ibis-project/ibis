SELECT t0.*
FROM (
  SELECT t1.`x` + 1 AS `x`
  FROM t t1
) t0
WHERE t0.`x` > 1