SELECT
  CASE
    WHEN (
      t0.continent = 'NA'
    )
    THEN 'North America'
    WHEN (
      t0.continent = 'SA'
    )
    THEN 'South America'
    WHEN (
      t0.continent = 'EU'
    )
    THEN 'Europe'
    WHEN (
      t0.continent = 'AF'
    )
    THEN 'Africa'
    WHEN (
      t0.continent = 'AS'
    )
    THEN 'Asia'
    WHEN (
      t0.continent = 'OC'
    )
    THEN 'Oceania'
    WHEN (
      t0.continent = 'AN'
    )
    THEN 'Antarctica'
    ELSE 'Unknown continent'
  END AS cont,
  SUM(t0.population) AS total_pop
FROM countries AS t0
GROUP BY
  1