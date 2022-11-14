SELECT *
FROM foo
WHERE `y` > (
  SELECT max(`x`) AS `max`
  FROM bar
)