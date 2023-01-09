SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`)
  FROM star1
  WHERE `foo_id` = 'foo'
)