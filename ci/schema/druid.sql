REPLACE INTO "diamonds"
OVERWRITE ALL
SELECT *
FROM TABLE(
  EXTERN(
    '{"type":"local","files":["/data/diamonds.parquet"]}',
    '{"type":"parquet"}',
    '[{"name":"carat","type":"double"},{"name":"cut","type":"string"},{"name":"color","type":"string"},{"name":"clarity","type":"string"},{"name":"depth","type":"double"},{"name":"table","type":"double"},{"name":"price","type":"long"},{"name":"x","type":"double"},{"name":"y","type":"double"},{"name":"z","type":"double"}]'
  )
)
PARTITIONED BY ALL TIME;

REPLACE INTO "batting"
OVERWRITE ALL
SELECT *
FROM TABLE(
  EXTERN(
    '{"type":"local","files":["/data/batting.parquet"]}',
    '{"type":"parquet"}',
    '[{"name":"playerID","type":"string"},{"name":"yearID","type":"long"},{"name":"stint","type":"long"},{"name":"teamID","type":"string"},{"name":"lgID","type":"string"},{"name":"G","type":"long"},{"name":"AB","type":"long"},{"name":"R","type":"long"},{"name":"H","type":"long"},{"name":"X2B","type":"long"},{"name":"X3B","type":"long"},{"name":"HR","type":"long"},{"name":"RBI","type":"long"},{"name":"SB","type":"long"},{"name":"CS","type":"long"},{"name":"BB","type":"long"},{"name":"SO","type":"long"},{"name":"IBB","type":"long"},{"name":"HBP","type":"long"},{"name":"SH","type":"long"},{"name":"SF","type":"long"},{"name":"GIDP","type":"long"}]'
  )
)
PARTITIONED BY ALL TIME;

REPLACE INTO "awards_players"
OVERWRITE ALL
SELECT *
FROM TABLE(
  EXTERN(
    '{"type":"local","files":["/data/awards_players.parquet"]}',
    '{"type":"parquet"}',
    '[{"name":"playerID","type":"string"},{"name":"awardID","type":"string"},{"name":"yearID","type":"long"},{"name":"lgID","type":"string"},{"name":"tie","type":"string"},{"name":"notes","type":"string"}]'
  )
)
PARTITIONED BY ALL TIME;

REPLACE INTO "functional_alltypes"
OVERWRITE ALL
SELECT *
FROM TABLE(
  EXTERN(
    '{"type":"local","files":["/data/functional_alltypes.parquet"]}',
    '{"type":"parquet"}',
    '[{"name":"id","type":"long"},{"name":"bool_col","type":"long"},{"name":"tinyint_col","type":"long"},{"name":"smallint_col","type":"long"},{"name":"int_col","type":"long"},{"name":"bigint_col","type":"long"},{"name":"float_col","type":"double"},{"name":"double_col","type":"double"},{"name":"date_string_col","type":"string"},{"name":"string_col","type":"string"},{"name":"timestamp_col","type":"string"},{"name":"year","type":"long"},{"name":"month","type":"long"}]'
  )
)
PARTITIONED BY ALL TIME;
