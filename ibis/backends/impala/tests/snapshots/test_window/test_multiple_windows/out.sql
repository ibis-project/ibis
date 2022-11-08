SELECT `g`, sum(`f`) OVER (PARTITION BY `g`) - sum(`f`) OVER () AS `result`
FROM alltypes