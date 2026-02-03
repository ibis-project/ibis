DROP TABLE IF EXISTS diamonds CASCADE;

CREATE TABLE diamonds (
    carat FLOAT,
    cut TEXT,
    color TEXT,
    clarity TEXT,
    depth FLOAT,
    "table" FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);

-- Note: In real usage, data would be loaded via sources or INSERT statements
-- For CI, we'll need to mount CSV files and use a different loading strategy

DROP TABLE IF EXISTS astronauts CASCADE;

CREATE TABLE astronauts (
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
);

DROP TABLE IF EXISTS batting CASCADE;

CREATE TABLE batting (
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
);

DROP TABLE IF EXISTS awards_players CASCADE;

CREATE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    "tie" TEXT,
    "notes" TEXT
);

DROP TABLE IF EXISTS functional_alltypes CASCADE;

CREATE TABLE functional_alltypes (
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
);

DROP TABLE IF EXISTS tzone CASCADE;

CREATE TABLE tzone (
    "ts" TIMESTAMP WITH TIME ZONE,
    "key" TEXT,
    "value" DOUBLE PRECISION
);

INSERT INTO tzone
    SELECT
	CAST('2017-05-28 11:01:31.000400' AS TIMESTAMP WITH TIME ZONE) +
	    (t * INTERVAL '1 day' + t * INTERVAL '1 second') AS "ts",
	CHR(97 + t) AS "key",
	t + t / 10.0 AS "value"
    FROM generate_series(0, 9) AS t;

DROP TABLE IF EXISTS array_types CASCADE;

CREATE TABLE IF NOT EXISTS array_types (
    "x" BIGINT[],
    "y" TEXT[],
    "z" DOUBLE PRECISION[],
    "grouper" TEXT,
    "scalar_column" DOUBLE PRECISION,
    "multi_dim" BIGINT[][]
);

-- Note: Materialize does not currently support multi-dimensional arrays containing NULL sub-arrays, so they are not included.
-- Fix pending: https://github.com/MaterializeInc/materialize/pull/33786

INSERT INTO array_types VALUES
    (ARRAY[1::BIGINT, 2::BIGINT, 3::BIGINT], ARRAY['a', 'b', 'c'], ARRAY[1.0::DOUBLE PRECISION, 2.0::DOUBLE PRECISION, 3.0::DOUBLE PRECISION], 'a', 1.0, NULL),
    (ARRAY[4::BIGINT, 5::BIGINT], ARRAY['d', 'e'], ARRAY[4.0::DOUBLE PRECISION, 5.0::DOUBLE PRECISION], 'a', 2.0, NULL),
    (ARRAY[6::BIGINT, NULL], ARRAY['f', NULL], ARRAY[6.0::DOUBLE PRECISION, NULL::DOUBLE PRECISION], 'a', 3.0, NULL),
    (ARRAY[NULL::BIGINT, 1::BIGINT, NULL::BIGINT], ARRAY[NULL, 'a', NULL], ARRAY[]::DOUBLE PRECISION[], 'b', 4.0, NULL),
    (ARRAY[2::BIGINT, NULL, 3::BIGINT], ARRAY['b', NULL, 'c'], NULL, 'b', 5.0, NULL),
    (ARRAY[4::BIGINT, NULL, NULL, 5::BIGINT], ARRAY['d', NULL, NULL, 'e'], ARRAY[4.0::DOUBLE PRECISION, NULL::DOUBLE PRECISION, NULL::DOUBLE PRECISION, 5.0::DOUBLE PRECISION], 'c', 6.0, NULL);

DROP TABLE IF EXISTS json_t CASCADE;

CREATE TABLE IF NOT EXISTS json_t (rowid BIGINT, "js" JSONB);

INSERT INTO json_t VALUES
    (1, '{"a": [1,2,3,4], "b": 1}'),
    (2, '{"a":null,"b":2}'),
    (3, '{"a":"foo", "c":null}'),
    (4, 'null'),
    (5, '[42,47,55]'),
    (6, '[]'),
    (7, '"a"'),
    (8, '""'),
    (9, '"b"'),
    (10, NULL),
    (11, 'true'),
    (12, 'false'),
    (13, '42'),
    (14, '37.37');

DROP TABLE IF EXISTS win CASCADE;

CREATE TABLE win ("g" TEXT, "x" BIGINT, "y" BIGINT);

INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

DROP TABLE IF EXISTS topk CASCADE;

CREATE TABLE topk ("x" BIGINT);

INSERT INTO topk VALUES (1), (1), (NULL);

DROP TABLE IF EXISTS map CASCADE;

CREATE TABLE map (idx BIGINT, kv JSONB);

INSERT INTO map VALUES
    (1, '{"a": 1, "b": 2, "c": 3}'),
    (2, '{"d": 4, "e": 5, "f": 6}');
