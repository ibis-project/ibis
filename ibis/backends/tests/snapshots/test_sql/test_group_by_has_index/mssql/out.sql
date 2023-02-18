SELECT
  CASE t0.continent
    WHEN N'NA'
    THEN N'North America'
    WHEN N'SA'
    THEN N'South America'
    WHEN N'EU'
    THEN N'Europe'
    WHEN N'AF'
    THEN N'Africa'
    WHEN N'AS'
    THEN N'Asia'
    WHEN N'OC'
    THEN N'Oceania'
    WHEN N'AN'
    THEN N'Antarctica'
    ELSE N'Unknown continent'
  END AS cont,
  SUM(t0.population) AS total_pop
FROM countries AS t0
GROUP BY
  CASE t0.continent
    WHEN N'NA'
    THEN N'North America'
    WHEN N'SA'
    THEN N'South America'
    WHEN N'EU'
    THEN N'Europe'
    WHEN N'AF'
    THEN N'Africa'
    WHEN N'AS'
    THEN N'Asia'
    WHEN N'OC'
    THEN N'Oceania'
    WHEN N'AN'
    THEN N'Antarctica'
    ELSE N'Unknown continent'
  END