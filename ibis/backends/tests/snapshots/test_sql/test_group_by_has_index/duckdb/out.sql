SELECT
  CASE
    WHEN (
      t0.continent = CAST('NA' AS TEXT)
    )
    THEN CAST('North America' AS TEXT)
    WHEN (
      t0.continent = CAST('SA' AS TEXT)
    )
    THEN CAST('South America' AS TEXT)
    WHEN (
      t0.continent = CAST('EU' AS TEXT)
    )
    THEN CAST('Europe' AS TEXT)
    WHEN (
      t0.continent = CAST('AF' AS TEXT)
    )
    THEN CAST('Africa' AS TEXT)
    WHEN (
      t0.continent = CAST('AS' AS TEXT)
    )
    THEN CAST('Asia' AS TEXT)
    WHEN (
      t0.continent = CAST('OC' AS TEXT)
    )
    THEN CAST('Oceania' AS TEXT)
    WHEN (
      t0.continent = CAST('AN' AS TEXT)
    )
    THEN CAST('Antarctica' AS TEXT)
    ELSE CAST('Unknown continent' AS TEXT)
  END AS cont,
  SUM(t0.population) AS total_pop
FROM countries AS t0
GROUP BY
  1