SELECT
  *
FROM (
  SELECT
    "t0"."station",
    "t0"."n"
  FROM "temp"."main"."pivot_fish" AS "t0"
)
PIVOT(COUNT("n") FOR "station" IN ('Lisbon' AS "Lisbon", 'Release' AS "Release"))