import argparse
import sys

import duckdb

# --stage restaurants rebuilds only restaurants.parquet and leaves
# reviews.parquet / users.parquet untouched.
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["all", "restaurants"], default="all")
args = parser.parse_args()

con = duckdb.connect()
con.execute("""
COPY (
  SELECT business_id, name, latitude, longitude,
         stars, review_count, categories, attributes,
         to_json(hours) AS hours,
         CAST(is_open AS BOOLEAN) AS is_open  -- reported only; closed restaurants are kept
  FROM read_json_auto('data/raw/yelp_academic_dataset_business.json')
  WHERE lower(trim(city)) IN (
    'philadelphia',
    'phila',
    'philadephia',
    'philadelphila',
    'philiidelphia',
    'philiadelphia',
    'philly',
    'philadelphia pa',
    'southwest philadelphia',
    'west philadelphia',
    'center city philadelphia',
    'philadelphia (northeast philly)'
  )
    AND categories LIKE '%Restaurants%'
    AND review_count >= 20
  -- Without ORDER BY, DuckDB's parallel scan writes rows in a different order
  -- on every run; sorted, the order is stable and matches business_ids.txt.
  ORDER BY business_id
) TO 'data/processed/restaurants.parquet'
""")

n_restaurants, n_with_hours, n_open, n_closed = con.execute("""
    SELECT count(*), count(hours), count(*) FILTER (is_open), count(*) FILTER (NOT is_open)
    FROM 'data/processed/restaurants.parquet'
""").fetchone()
print(f"restaurants: {n_restaurants} rows, {n_with_hours} with hours, {n_open} open, {n_closed} closed")

if args.stage == "restaurants":
    sys.exit(0)

con.execute("""
COPY (
  SELECT r.review_id, r.user_id, r.business_id,
         r.stars, r.text, r.date
  FROM read_json_auto('data/raw/yelp_academic_dataset_review.json') r
  JOIN 'data/processed/restaurants.parquet' b
    ON r.business_id = b.business_id
) TO 'data/processed/reviews.parquet'
""")

print(con.execute(
    "SELECT count(*) FROM 'data/processed/reviews.parquet'"
).fetchone())

con.execute("""
COPY (
  SELECT user_id, count(*) AS n_reviews, avg(stars) AS avg_stars
  FROM 'data/processed/reviews.parquet'
  GROUP BY user_id
  HAVING count(*) >= 5
) TO 'data/processed/users.parquet'
""")

print(con.execute("""
    SELECT count(*) AS n_users, sum(n_reviews) AS n_reviews
    FROM 'data/processed/users.parquet'
""").fetchone())