SELECT
  CASE t0."continent"
    WHEN 'NA'
    THEN 'North America'
    WHEN 'SA'
    THEN 'South America'
    WHEN 'EU'
    THEN 'Europe'
    WHEN 'AF'
    THEN 'Africa'
    WHEN 'AS'
    THEN 'Asia'
    WHEN 'OC'
    THEN 'Oceania'
    WHEN 'AN'
    THEN 'Antarctica'
    ELSE 'Unknown continent'
  END AS "cont",
  SUM(t0."population") AS "total_pop"
FROM "countries" t0
GROUP BY
  CASE t0."continent"
    WHEN 'NA'
    THEN 'North America'
    WHEN 'SA'
    THEN 'South America'
    WHEN 'EU'
    THEN 'Europe'
    WHEN 'AF'
    THEN 'Africa'
    WHEN 'AS'
    THEN 'Asia'
    WHEN 'OC'
    THEN 'Oceania'
    WHEN 'AN'
    THEN 'Antarctica'
    ELSE 'Unknown continent'
  END