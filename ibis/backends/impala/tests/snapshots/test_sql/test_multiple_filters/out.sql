SELECT *
FROM (
  SELECT *
  FROM t0
  WHERE `a` < 100
) t0
WHERE `a` = (
  SELECT max(`a`) AS `max`
  FROM t0
  WHERE `a` < 100
)