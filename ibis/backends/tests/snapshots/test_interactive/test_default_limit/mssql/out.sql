SELECT
  [t2].[id],
  IIF([t2].[bool_col] <> 0, 1, 0) AS [bool_col]
FROM (
  SELECT
  TOP 11
    [t0].[id],
    [t0].[bool_col]
  FROM [functional_alltypes] AS [t0]
) AS [t2]