SELECT lag(t0.`f` - lag(t0.`f`) OVER (ORDER BY t0.`f` ASC)) OVER (ORDER BY t0.`f` ASC) AS `foo`
FROM `alltypes` t0