SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`) AS `Mean(f)`
  FROM star1
  WHERE `foo_id` = 'foo'
)