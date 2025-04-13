SELECT
  [x],
  [y]
FROM (
  SELECT
    [t1].[x] AS [x],
    [t1].[y] AS [y],
    AVG([t1].[x]) OVER (
      ORDER BY CASE WHEN [t1].[x] IS NULL THEN 1 ELSE 0 END, [t1].[x] ASC
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS _w
  FROM (
    SELECT
      [t0].[x],
      SUM([t0].[x]) OVER (
        ORDER BY CASE WHEN [t0].[x] IS NULL THEN 1 ELSE 0 END, [t0].[x] ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
      ) AS [y]
    FROM [t] AS [t0]
  ) AS [t1]
  WHERE
    [t1].[y] <= 37
) AS _t
WHERE
  _w IS NOT NULL