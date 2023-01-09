SELECT *
FROM (
  SELECT *
  FROM my_table
  WHERE `a` < 100
) t0
WHERE (`a` = (
  SELECT max(`a`)
  FROM my_table
  WHERE `a` < 100
)) AND
      (`b` = 'a')