SELECT
  t1.key AS key,
  t2.key AS key_right,
  t4.key_right AS key_right_right
FROM (
  SELECT
    *
  FROM "leaf" AS t0
  WHERE
    TRUE
) AS t1
INNER JOIN (
  SELECT
    t1.key AS key
  FROM (
    SELECT
      *
    FROM "leaf" AS t0
    WHERE
      TRUE
  ) AS t1
) AS t2
  ON t1.key = t2.key
INNER JOIN (
  SELECT
    t1.key AS key,
    t2.key AS key_right
  FROM (
    SELECT
      *
    FROM "leaf" AS t0
    WHERE
      TRUE
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key AS key
    FROM (
      SELECT
        *
      FROM "leaf" AS t0
      WHERE
        TRUE
    ) AS t1
  ) AS t2
    ON t1.key = t2.key
) AS t4
  ON t1.key = t1.key