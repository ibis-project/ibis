SELECT t0.*
FROM star1 t0
WHERE t0.`f` > ln((
  SELECT avg(t0.`f`) AS `Mean(f)`
  FROM star1 t0
  WHERE t0.`foo_id` = 'foo'
))