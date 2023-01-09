SELECT *
FROM star1
WHERE `f` > ln((
  SELECT avg(`f`)
  FROM star1
  WHERE `foo_id` = 'foo'
))