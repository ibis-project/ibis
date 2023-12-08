SELECT
  t2.key AS key,
  t3.key AS key_right,
  t6.key_right AS key_right_right
FROM (
  SELECT
    t0.key AS key
  FROM leaf AS t0
  WHERE
    TRUE
) AS t2
INNER JOIN (
  SELECT
    t0.key AS key
  FROM leaf AS t0
  WHERE
    TRUE
) AS t3
  ON t2.key = t3.key
INNER JOIN (
  SELECT
    t2.key AS key,
    t3.key AS key_right
  FROM (
    SELECT
      t0.key AS key
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t2
  INNER JOIN (
    SELECT
      t0.key AS key
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t3
    ON t2.key = t3.key
) AS t6
  ON t6.key = t6.key