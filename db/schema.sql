CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS restaurants (
  business_id text PRIMARY KEY,
  name text NOT NULL,
  latitude double precision,
  longitude double precision,
  stars real,
  review_count int,
  price_level smallint,        -- 1-4 from attributes.RestaurantsPriceRange2, NULL if missing
  categories text[],
  attributes jsonb,
  hours jsonb,
  embedding vector(384) NOT NULL
);

CREATE OR REPLACE FUNCTION haversine_km(
  lat1 double precision, lon1 double precision,
  lat2 double precision, lon2 double precision
) RETURNS double precision
LANGUAGE sql IMMUTABLE AS $$
  SELECT 2 * 6371.0 * asin(sqrt(
    power(sin(radians(lat2 - lat1) / 2), 2) +
    cos(radians(lat1)) * cos(radians(lat2)) *
    power(sin(radians(lon2 - lon1) / 2), 2)
  ));
$$;
