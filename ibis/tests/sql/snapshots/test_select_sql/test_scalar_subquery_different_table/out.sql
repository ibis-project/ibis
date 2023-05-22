SELECT t0.*
FROM foo t0
WHERE t0.`y` > (
  SELECT max(t1.`x`) AS `Max(x)`
  FROM bar t1
)