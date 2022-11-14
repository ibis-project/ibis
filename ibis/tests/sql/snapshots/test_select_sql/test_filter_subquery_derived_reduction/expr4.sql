SELECT *
FROM star1
WHERE `f` > (ln((
  SELECT avg(`f`) AS `mean`
  FROM star1
  WHERE `foo_id` = 'foo'
)) + 1)