import duckdb

con = duckdb.connect()
con.execute("""
COPY (
  SELECT business_id, name, latitude, longitude,
         stars, review_count, categories, attributes
  FROM read_json_auto('data/raw/yelp_academic_dataset_business.json')
  WHERE city = 'Philadelphia'
    AND categories LIKE '%Restaurants%'
    AND review_count >= 20
) TO 'data/processed/restaurants.parquet'
""")

print(con.execute("SELECT count(*) FROM 'data/processed/restaurants.parquet'").fetchone())  

print(con.execute("""
    SELECT name, stars, review_count, categories
    FROM 'data/processed/restaurants.parquet'
    ORDER BY random() LIMIT 20
""").df().to_string())

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