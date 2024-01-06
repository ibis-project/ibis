SELECT
  IIF(
    [t0].[x] IN (
      SELECT
        [t0].[x]
      FROM [t] AS [t0]
      WHERE
        [t0].[x] > 2
    ),
    1,
    0
  ) AS [InSubquery(x)]
FROM [t] AS [t0]