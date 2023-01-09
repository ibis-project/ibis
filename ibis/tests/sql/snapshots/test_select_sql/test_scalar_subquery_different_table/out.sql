SELECT *
FROM foo
WHERE `y` > (
  SELECT max(`x`)
  FROM bar
)