CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS first_last_agg;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

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

COPY diamonds FROM '/data/diamonds.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

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

COPY astronauts FROM '/data/astronauts.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

DROP TABLE IF EXISTS batting CASCADE;

CREATE TABLE batting (
    "playerID" TEXT,
    "yearID" BIGINT,
    stint BIGINT,
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

COPY batting FROM '/data/batting.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

DROP TABLE IF EXISTS awards_players CASCADE;

CREATE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    tie TEXT,
    notes TEXT
);

COPY awards_players FROM '/data/awards_players.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

DROP VIEW IF EXISTS awards_players_special_types CASCADE;
CREATE VIEW awards_players_special_types AS
SELECT
    *,
    setweight(to_tsvector('simple', notes), 'A')::TSVECTOR AS search,
    '[1,2,3]'::VECTOR AS simvec
FROM awards_players;

DROP TABLE IF EXISTS functional_alltypes CASCADE;

CREATE TABLE functional_alltypes (
    id INTEGER,
    bool_col BOOLEAN,
    tinyint_col SMALLINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col REAL,
    double_col DOUBLE PRECISION,
    date_string_col TEXT,
    string_col TEXT,
    timestamp_col TIMESTAMP WITHOUT TIME ZONE,
    year INTEGER,
    month INTEGER
);

COPY functional_alltypes FROM '/data/functional_alltypes.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

DROP TABLE IF EXISTS tzone CASCADE;

CREATE TABLE tzone (
    ts TIMESTAMP WITH TIME ZONE,
    key TEXT,
    value DOUBLE PRECISION
);

INSERT INTO tzone
    SELECT
	CAST('2017-05-28 11:01:31.000400' AS TIMESTAMP WITH TIME ZONE) +
	    t * INTERVAL '1 day 1 microsecond' AS ts,
	CHR(97 + t) AS key,
	t + t / 10.0 AS value
    FROM generate_series(0, 9) AS t;

DROP TABLE IF EXISTS array_types CASCADE;

CREATE TABLE IF NOT EXISTS array_types (
    x BIGINT[],
    y TEXT[],
    z DOUBLE PRECISION[],
    grouper TEXT,
    scalar_column DOUBLE PRECISION,
    multi_dim BIGINT[][]
);

INSERT INTO array_types VALUES
    (ARRAY[1, 2, 3], ARRAY['a', 'b', 'c'], ARRAY[1.0, 2.0, 3.0], 'a', 1.0, ARRAY[ARRAY[NULL::BIGINT, NULL, NULL], ARRAY[1, 2, 3]]),
    (ARRAY[4, 5], ARRAY['d', 'e'], ARRAY[4.0, 5.0], 'a', 2.0, ARRAY[]::BIGINT[][]),
    (ARRAY[6, NULL], ARRAY['f', NULL], ARRAY[6.0, NULL], 'a', 3.0, ARRAY[NULL, ARRAY[]::BIGINT[], NULL]),
    (ARRAY[NULL, 1, NULL], ARRAY[NULL, 'a', NULL], ARRAY[]::DOUBLE PRECISION[], 'b', 4.0, ARRAY[ARRAY[1], ARRAY[2], ARRAY[NULL::BIGINT], ARRAY[3]]),
    (ARRAY[2, NULL, 3], ARRAY['b', NULL, 'c'], NULL, 'b', 5.0, NULL),
    (ARRAY[4, NULL, NULL, 5], ARRAY['d', NULL, NULL, 'e'], ARRAY[4.0, NULL, NULL, 5.0], 'c', 6.0, ARRAY[ARRAY[1, 2, 3]]);

DROP TABLE IF EXISTS films CASCADE;

CREATE TABLE IF NOT EXISTS films (
    code CHAR(5) PRIMARY KEY,
    title VARCHAR(40) NOT NULL,
    did INTEGER NOT NULL,
    date_prod DATE,
    kind VARCHAR(10),
    len INTERVAL HOUR TO MINUTE
);

INSERT INTO films VALUES
    ('A', 'Avengers', 1, DATE '2018-01-01', 'Action', INTERVAL '2 hours 35 minutes'),
    ('B', 'Ghostbusters', 2, DATE '2018-01-02', 'Ghost', INTERVAL '1 hour 30 minutes');

DROP TABLE IF EXISTS intervals CASCADE;

CREATE TABLE IF NOT EXISTS intervals (
    -- enable year and month when relativedelta support lands
    c INTERVAL DAY,
    d INTERVAL HOUR,
    e INTERVAL MINUTE,
    f INTERVAL SECOND,
    h INTERVAL DAY TO HOUR,
    i INTERVAL DAY TO MINUTE,
    j INTERVAL DAY TO SECOND,
    k INTERVAL HOUR TO MINUTE,
    l INTERVAL HOUR TO SECOND,
    m INTERVAL MINUTE TO SECOND
);

INSERT INTO intervals VALUES
    (
	-- '1 year',
	-- '1 month',
	'1 day',
	'1 hour',
	'-1 minute',
	'1 second 30 milliseconds -10 microseconds',
	-- '-1 year 3 months',
	'1 day 4 hours',
	'1 day 17 minutes',
	'-1 day -2 hours 17 minutes 30 seconds',
	'1 hour 2 minutes',
	'1 hour 2 minutes -7 seconds 37 microseconds',
	'1 minute 3 seconds 2 milliseconds 9 microseconds'
    );


CREATE TABLE IF NOT EXISTS not_supported_intervals (
    a INTERVAL YEAR,
    b INTERVAL YEAR,
    g INTERVAL YEAR TO MONTH
);

DROP TABLE IF EXISTS geo CASCADE;

CREATE TABLE geo (
    id BIGSERIAL PRIMARY KEY,
    geo_point GEOMETRY(POINT),
    geo_linestring GEOMETRY(LINESTRING),
    geo_polygon GEOMETRY(POLYGON),
    geo_multipolygon GEOMETRY(MULTIPOLYGON)
);

COPY geo FROM '/data/geo.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

CREATE INDEX IF NOT EXISTS idx_geo_geo_linestring ON geo USING GIST (geo_linestring);
CREATE INDEX IF NOT EXISTS idx_geo_geo_multipolygon ON geo USING GIST (geo_multipolygon);
CREATE INDEX IF NOT EXISTS idx_geo_geo_point ON geo USING GIST (geo_point);
CREATE INDEX IF NOT EXISTS idx_geo_geo_polygon ON geo USING GIST (geo_polygon);

DROP TABLE IF EXISTS json_t CASCADE;

CREATE TABLE IF NOT EXISTS json_t (js JSON);

INSERT INTO json_t VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');

DROP TABLE IF EXISTS win CASCADE;
CREATE TABLE win (g TEXT, x BIGINT NOT NULL, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

DROP TABLE IF EXISTS map CASCADE;
CREATE TABLE map (idx BIGINT, kv HSTORE);
INSERT INTO map VALUES
    (1, 'a=>1,b=>2,c=>3'),
    (2, 'd=>4,e=>5,c=>6');
