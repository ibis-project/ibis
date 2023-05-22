SELECT lag(t0.`d`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` DESC) AS `foo`,
       max(t0.`a`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` DESC) AS `Max(a)`
FROM `alltypes` t0