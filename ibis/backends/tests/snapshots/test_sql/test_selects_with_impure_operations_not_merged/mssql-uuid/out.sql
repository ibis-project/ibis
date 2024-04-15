SELECT
  [t1].[x],
  [t1].[y],
  [t1].[z],
  IIF([t1].[y] = [t1].[z], 'big', 'small') AS [size]
FROM (
  SELECT
    [t0].[x],
    NEWID() AS [y],
    NEWID() AS [z]
  FROM [t] AS [t0]
) AS [t1]