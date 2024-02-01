DROP TABLE IF EXISTS hive.default.diamonds;
CREATE TABLE hive.default.diamonds (
    "carat" DOUBLE,
    "cut" VARCHAR,
    "color" VARCHAR,
    "clarity" VARCHAR,
    "depth" DOUBLE,
    "table" DOUBLE,
    "price" BIGINT,
    "x" DOUBLE,
    "y" DOUBLE,
    "z" DOUBLE
) WITH (
    external_location = 's3a://trino/diamonds',
    format = 'PARQUET'
);

CREATE OR REPLACE VIEW memory.default.diamonds AS
SELECT * FROM hive.default.diamonds;

DROP TABLE IF EXISTS hive.default.astronauts;
CREATE TABLE hive.default.astronauts (
    "id" BIGINT,
    "number" BIGINT,
    "nationwide_number" BIGINT,
    "name" VARCHAR,
    "original_name" VARCHAR,
    "sex" VARCHAR,
    "year_of_birth" BIGINT,
    "nationality" VARCHAR,
    "military_civilian" VARCHAR,
    "selection" VARCHAR,
    "year_of_selection" BIGINT,
    "mission_number" BIGINT,
    "total_number_of_missions" BIGINT,
    "occupation" VARCHAR,
    "year_of_mission" BIGINT,
    "mission_title" VARCHAR,
    "ascend_shuttle" VARCHAR,
    "in_orbit" VARCHAR,
    "descend_shuttle" VARCHAR,
    "hours_mission" REAL,
    "total_hrs_sum" REAL,
    "field21" BIGINT,
    "eva_hrs_mission" REAL,
    "total_eva_hrs" REAL
) WITH (
    external_location = 's3a://trino/astronauts',
    format = 'PARQUET'
);

CREATE OR REPLACE VIEW memory.default.astronauts AS
SELECT * FROM hive.default.astronauts;

DROP TABLE IF EXISTS hive.default.batting;
CREATE TABLE hive.default.batting (
    "playerID" VARCHAR,
    "yearID" BIGINT,
    "stint" BIGINT,
    "teamID" VARCHAR,
    "lgID" VARCHAR,
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
) WITH (
    external_location = 's3a://trino/batting',
    format = 'PARQUET'
);

CREATE OR REPLACE VIEW memory.default.batting AS
SELECT * FROM hive.default.batting;

DROP TABLE IF EXISTS hive.default.awards_players;
CREATE TABLE hive.default.awards_players (
    "playerID" VARCHAR,
    "awardID" VARCHAR,
    "yearID" BIGINT,
    "lgID" VARCHAR,
    "tie" VARCHAR,
    "notes" VARCHAR
) WITH (
    external_location = 's3a://trino/awards-players',
    format = 'PARQUET'
);

CREATE OR REPLACE VIEW memory.default.awards_players AS
SELECT * FROM hive.default.awards_players;

DROP TABLE IF EXISTS hive.default.functional_alltypes;
CREATE TABLE hive.default.functional_alltypes (
    "id" INTEGER,
    "bool_col" BOOLEAN,
    "tinyint_col" TINYINT,
    "smallint_col" SMALLINT,
    "int_col" INTEGER,
    "bigint_col" BIGINT,
    "float_col" REAL,
    "double_col" DOUBLE,
    "date_string_col" VARCHAR,
    "string_col" VARCHAR,
    "timestamp_col" TIMESTAMP(6),
    "year" INTEGER,
    "month" INTEGER
) WITH (
    external_location = 's3a://trino/functional-alltypes',
    format = 'PARQUET'
);
CREATE OR REPLACE VIEW memory.default.functional_alltypes AS
SELECT * FROM hive.default.functional_alltypes;

DROP TABLE IF EXISTS array_types;

CREATE TABLE IF NOT EXISTS array_types (
    x ARRAY<BIGINT>,
    y ARRAY<VARCHAR>,
    z ARRAY<DOUBLE>,
    grouper VARCHAR,
    scalar_column DOUBLE,
    multi_dim ARRAY<ARRAY<BIGINT>>
);

INSERT INTO array_types VALUES
    (ARRAY[1, 2, 3], ARRAY['a', 'b', 'c'], ARRAY[1.0, 2.0, 3.0], 'a', 1.0, ARRAY[ARRAY[NULL, NULL, NULL], ARRAY[1, 2, 3]]),
    (ARRAY[4, 5], ARRAY['d', 'e'], ARRAY[4.0, 5.0], 'a', 2.0, ARRAY[]),
    (ARRAY[6, NULL], ARRAY['f', NULL], ARRAY[6.0, NULL], 'a', 3.0, ARRAY[NULL, ARRAY[], NULL]),
    (ARRAY[NULL, 1, NULL], ARRAY[NULL, 'a', NULL], ARRAY[], 'b', 4.0, ARRAY[ARRAY[1], ARRAY[2], ARRAY[NULL], ARRAY[3]]),
    (ARRAY[2, NULL, 3], ARRAY['b', NULL, 'c'], NULL, 'b', 5.0, NULL),
    (ARRAY[4, NULL, NULL, 5], ARRAY['d', NULL, NULL, 'e'], ARRAY[4.0, NULL, NULL, 5.0], 'c', 6.0, ARRAY[ARRAY[1, 2, 3]]);

DROP TABLE IF EXISTS map;
CREATE TABLE map (idx BIGINT, kv MAP<VARCHAR, BIGINT>);
INSERT INTO map VALUES
    (1, MAP(ARRAY['a', 'b', 'c'], ARRAY[1, 2, 3])),
    (2, MAP(ARRAY['d', 'e', 'f'], ARRAY[4, 5, 6]));

DROP TABLE IF EXISTS struct;
CREATE TABLE struct (abc ROW(a DOUBLE, b VARCHAR, c BIGINT));
INSERT INTO struct
    SELECT ROW(1.0, 'banana', 2) UNION
    SELECT ROW(2.0, 'apple', 3) UNION
    SELECT ROW(3.0, 'orange', 4) UNION
    SELECT ROW(NULL, 'banana', 2) UNION
    SELECT ROW(2.0, NULL, 3) UNION
    SELECT NULL UNION
    SELECT ROW(3.0, 'orange', NULL);

DROP TABLE IF EXISTS memory.default.json_t;

CREATE TABLE IF NOT EXISTS memory.default.json_t (js JSON);

INSERT INTO memory.default.json_t VALUES
    (JSON '{"a": [1,2,3,4], "b": 1}'),
    (JSON '{"a":null,"b":2}'),
    (JSON '{"a":"foo", "c":null}'),
    (JSON 'null'),
    (JSON '[42,47,55]'),
    (JSON '[]');

DROP TABLE IF EXISTS win;
CREATE TABLE win (g VARCHAR, x BIGINT, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
