SELECT
  [t0].[color],
  [t0].[price],
  RANK() OVER (PARTITION BY [t0].[color] ORDER BY CASE WHEN [t0].[price] IS NULL THEN 1 ELSE 0 END, [t0].[price] ASC) - 1 AS [MinRank()]
FROM [diamonds_sample] AS [t0]