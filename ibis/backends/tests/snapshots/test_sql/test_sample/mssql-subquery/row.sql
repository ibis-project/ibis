SELECT
  [t0].[x],
  [t0].[y]
FROM [test] AS [t0]
WHERE
  RAND(CHECKSUM(NEWID())) <= 0.5