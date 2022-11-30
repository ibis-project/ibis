SELECT *
FROM (
  SELECT *
  FROM t0
  WHERE `a` < 100
) t0
WHERE `a` = (
  SELECT max(`a`) AS `Max(a)`
  FROM t0
  WHERE `a` < 100
)