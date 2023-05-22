SELECT t0.*
FROM star1 t0
WHERE t0.`f` > (
  SELECT avg(t0.`f`) AS `Mean(f)`
  FROM star1 t0
)