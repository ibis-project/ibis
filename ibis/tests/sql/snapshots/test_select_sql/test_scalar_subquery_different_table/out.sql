SELECT *
FROM foo
WHERE `y` > (
  SELECT max(`x`) AS `Max(x)`
  FROM bar
)