SELECT
  *
FROM (
  SELECT
    *
  FROM "test" "t0"
  WHERE
    "t0"."x" > 10
) "t1"
WHERE
  DBMS_RANDOM.VALUE() <= 0.5