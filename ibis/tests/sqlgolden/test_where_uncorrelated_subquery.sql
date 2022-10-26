SELECT *
FROM foo
WHERE `job` IN (
  SELECT `job`
  FROM bar
)
