# rnjsjffoodw

A restaurant recommender for Philadelphia built on the Yelp Academic Dataset.

restaurants.parquet
  business_id (str), name (str), latitude (float), longitude (float),
  stars (float), review_count (int), categories (str), price_level (int, 1-4)

reviews.parquet
  review_id (str), user_id (str), business_id (str),
  stars (int), text (str), date (timestamp)

embeddings.npy    -> (n_restaurants, 384) float32
business_ids.txt  -> one business_id per line, same order as embeddings