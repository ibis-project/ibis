SELECT
  [t1].[x],
  [t1].[y]
FROM (
  SELECT
    [t0].[x],
    [t0].[y]
  FROM [test] AS [t0]
  WHERE
    [t0].[x] > 10
) AS [t1]
WHERE
  RAND(CHECKSUM(NEWID())) <= 0.5