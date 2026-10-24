-- Group-API tables for a hosted (DATA_SOURCE=db) deployment (Person A).
-- Filled by `python -m src.deploy_load`; db/schema.sql's restaurants table holds the
-- restaurant metadata and raw embeddings. No review text is stored anywhere.
-- Idempotent: deploy_load applies it on every run.
CREATE EXTENSION IF NOT EXISTS vector;

-- ContentBasedDebiased taste vectors (whole train history, centered embedding space)
-- for the demo users only.
CREATE TABLE IF NOT EXISTS user_taste (
  user_id text PRIMARY KEY,
  embedding vector(384) NOT NULL
);

-- Fitted model arrays that aren't per user. 'hub': ContentBasedDebiased's per-restaurant
-- mean score over all served users (5+ reviews), in business_id byte order.
CREATE TABLE IF NOT EXISTS model_arrays (
  name text PRIMARY KEY,
  model text NOT NULL,
  n integer NOT NULL,
  vals real[] NOT NULL
);

-- Each demo user's top-5 train restaurants by stars (ties by business_id): the
-- "profile" shown in the UI. Ratings only, no review text.
CREATE TABLE IF NOT EXISTS user_profiles (
  user_id text NOT NULL,
  rank smallint NOT NULL,
  business_id text NOT NULL,
  stars real NOT NULL,
  PRIMARY KEY (user_id, rank)
);

-- Every restaurant each demo user reviewed in train (IDs only), for exclude_visited.
CREATE TABLE IF NOT EXISTS user_visited (
  user_id text NOT NULL,
  business_id text NOT NULL,
  PRIMARY KEY (user_id, business_id)
);

-- Val groups drawn from the demo users, for GET /group/users/sample.
CREATE TABLE IF NOT EXISTS sample_groups (
  group_id text PRIMARY KEY,
  members text[] NOT NULL,
  cohesion real NOT NULL,
  cohesion_bucket text NOT NULL CHECK (cohesion_bucket IN ('high', 'low'))
);
