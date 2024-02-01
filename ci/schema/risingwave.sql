SET RW_IMPLICIT_FLUSH=true;

DROP TABLE IF EXISTS "diamonds" CASCADE;

CREATE TABLE "diamonds" (
    "carat" FLOAT,
    "cut" TEXT,
    "color" TEXT,
    "clarity" TEXT,
    "depth" FLOAT,
    "table" FLOAT,
    "price" BIGINT,
    "x" FLOAT,
    "y" FLOAT,
    "z" FLOAT
) WITH (
  connector = 'posix_fs',
  match_pattern = 'diamonds.csv',
  posix_fs.root = '/data',
) FORMAT PLAIN ENCODE CSV ( without_header = 'false', delimiter = ',' );

DROP TABLE IF EXISTS "astronauts" CASCADE;

CREATE TABLE "astronauts" (
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
    "hours_mission" DOUBLE PRECISION,
    "total_hrs_sum" DOUBLE PRECISION,
    "field21" BIGINT,
    "eva_hrs_mission" DOUBLE PRECISION,
    "total_eva_hrs" DOUBLE PRECISION
) WITH (
  connector = 'posix_fs',
  match_pattern = 'astronauts.csv',
  posix_fs.root = '/data',
) FORMAT PLAIN ENCODE CSV ( without_header = 'false', delimiter = ',' );

DROP TABLE IF EXISTS "batting" CASCADE;

CREATE TABLE "batting" (
    "playerID" TEXT,
    "yearID" BIGINT,
    "stint" BIGINT,
    "teamID" TEXT,
    "lgID" TEXT,
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
  connector = 'posix_fs',
  match_pattern = 'batting.csv',
  posix_fs.root = '/data',
) FORMAT PLAIN ENCODE CSV ( without_header = 'false', delimiter = ',' );

DROP TABLE IF EXISTS "awards_players" CASCADE;

CREATE TABLE "awards_players" (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    "tie" TEXT,
    "notes" TEXT
) WITH (
  connector = 'posix_fs',
  match_pattern = 'awards_players.csv',
  posix_fs.root = '/data',
) FORMAT PLAIN ENCODE CSV ( without_header = 'false', delimiter = ',' );

DROP TABLE IF EXISTS "functional_alltypes" CASCADE;

CREATE TABLE "functional_alltypes" (
    "id" INTEGER,
    "bool_col" BOOLEAN,
    "tinyint_col" SMALLINT,
    "smallint_col" SMALLINT,
    "int_col" INTEGER,
    "bigint_col" BIGINT,
    "float_col" REAL,
    "double_col" DOUBLE PRECISION,
    "date_string_col" TEXT,
    "string_col" TEXT,
    "timestamp_col" TIMESTAMP WITHOUT TIME ZONE,
    "year" INTEGER,
    "month" INTEGER
) WITH (
  connector = 'posix_fs',
  match_pattern = 'functional_alltypes.csv',
  posix_fs.root = '/data',
) FORMAT PLAIN ENCODE CSV ( without_header = 'false', delimiter = ',' );

DROP TABLE IF EXISTS "tzone" CASCADE;

CREATE TABLE "tzone" (
    "ts" TIMESTAMP WITH TIME ZONE,
    "key" TEXT,
    "value" DOUBLE PRECISION
);

INSERT INTO "tzone"
    SELECT
	CAST('2017-05-28 11:01:31.000400' AS TIMESTAMP WITH TIME ZONE) +
	    t * INTERVAL '1 day 1 second' AS "ts",
	CHR(97 + t) AS "key",
	t + t / 10.0 AS "value"
    FROM generate_series(0, 9) AS "t";

DROP TABLE IF EXISTS "array_types" CASCADE;

CREATE TABLE IF NOT EXISTS "array_types" (
    "x" BIGINT[],
    "y" TEXT[],
    "z" DOUBLE PRECISION[],
    "grouper" TEXT,
    "scalar_column" DOUBLE PRECISION,
    "multi_dim" BIGINT[][]
);

INSERT INTO "array_types" VALUES
    (ARRAY[1, 2, 3], ARRAY['a', 'b', 'c'], ARRAY[1.0, 2.0, 3.0], 'a', 1.0, ARRAY[ARRAY[NULL::BIGINT, NULL, NULL], ARRAY[1, 2, 3]]),
    (ARRAY[4, 5], ARRAY['d', 'e'], ARRAY[4.0, 5.0], 'a', 2.0, ARRAY[]::BIGINT[][]),
    (ARRAY[6, NULL], ARRAY['f', NULL], ARRAY[6.0, NULL], 'a', 3.0, ARRAY[NULL, ARRAY[]::BIGINT[], NULL]),
    (ARRAY[NULL, 1, NULL], ARRAY[NULL, 'a', NULL], ARRAY[]::DOUBLE PRECISION[], 'b', 4.0, ARRAY[ARRAY[1], ARRAY[2], ARRAY[NULL::BIGINT], ARRAY[3]]),
    (ARRAY[2, NULL, 3], ARRAY['b', NULL, 'c'], NULL, 'b', 5.0, NULL),
    (ARRAY[4, NULL, NULL, 5], ARRAY['d', NULL, NULL, 'e'], ARRAY[4.0, NULL, NULL, 5.0], 'c', 6.0, ARRAY[ARRAY[1, 2, 3]]);

DROP TABLE IF EXISTS "json_t" CASCADE;

CREATE TABLE IF NOT EXISTS "json_t" ("js" JSONB);

INSERT INTO "json_t" VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');

DROP TABLE IF EXISTS "win" CASCADE;
CREATE TABLE "win" ("g" TEXT, "x" BIGINT, "y" BIGINT);
INSERT INTO "win" VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
