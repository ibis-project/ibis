SELECT
  COUNT(*) AS `CountStar()`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `b`
    FROM (VALUES
      (1),
      (2)) AS `mytable`(`b`)
  ) AS `t0`
) AS `t1`