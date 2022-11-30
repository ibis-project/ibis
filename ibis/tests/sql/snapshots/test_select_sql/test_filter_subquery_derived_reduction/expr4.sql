SELECT *
FROM star1
WHERE `f` > (ln((
  SELECT avg(`f`) AS `Mean(f)`
  FROM star1
  WHERE `foo_id` = 'foo'
)) + 1)