SELECT
  *
FROM events AS t0
WHERE
  EXISTS(
    (
      SELECT
        CAST(1 AS TINYINT) AS "1"
      FROM (
        SELECT
          *
        FROM purchases AS t1
        WHERE
          (
            t1.ts > '2015-08-15'
          ) AND (
            t0.user_id = t1.user_id
          )
      ) AS t2
    )
  )