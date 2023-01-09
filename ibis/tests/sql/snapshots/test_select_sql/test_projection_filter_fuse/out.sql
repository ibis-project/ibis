SELECT `a` AS `a`, `b` AS `b`, `c` AS `c`
FROM (
  SELECT *
  FROM foo
  WHERE `a` > 0
) t0