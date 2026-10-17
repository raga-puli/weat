-- Group-recommendation tables (Person A), alongside db/schema.sql's restaurants table.
-- Not mounted by docker-compose; `python -m src.db_groups load-taste` applies it (idempotent)
-- and fills all three tables in one transaction.
CREATE EXTENSION IF NOT EXISTS vector;

-- ContentBasedCentered's catalog: embeddings minus the catalog mean, L2-normalized.
-- (restaurants.embedding holds the raw, uncentered vectors.)
CREATE TABLE IF NOT EXISTS restaurant_centered (
  business_id text PRIMARY KEY,
  embedding vector(384) NOT NULL
);

-- ContentBasedCentered taste vectors (fit on train) for every user with 5+ reviews.
CREATE TABLE IF NOT EXISTS user_taste (
  user_id text PRIMARY KEY,
  embedding vector(384) NOT NULL
);

-- Restaurants each of those users reviewed in train, for exclude_reviewed.
-- B's schema has no reviews table, so this lives here.
CREATE TABLE IF NOT EXISTS user_reviewed (
  user_id text NOT NULL,
  business_id text NOT NULL,
  PRIMARY KEY (user_id, business_id)
);
