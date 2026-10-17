# weat

<img src="assets/weat.jpg" width="70%">


A restaurant recommendation system built on Yelp-style business, review, and embedding data.

The app is called WEat — a combination of "we" and "eat".

## Counts

| metric | value |
| --- | --- |
| restaurants (review_count >= 20) | 3902 |
| their reviews | 666849 |
| users (5+ reviews) | 26093 |
| those users' reviews | 399881 |

restaurants.parquet
  business_id (str), name (str), latitude (float), longitude (float),
  stars (float), review_count (int), categories (str), price_level (int, 1-4),
  hours (JSON str): object keyed by weekday, values "H:M-H:M" (minutes not zero-padded);
    days the restaurant is closed or unlisted are null; whole field null for 318 restaurants with no hours

reviews.parquet
  review_id (str), user_id (str), business_id (str),
  stars (int), text (str), date (timestamp)

embeddings.npy    -> (n_restaurants, 384) float32
business_ids.txt  -> one business_id per line, same order as embeddings

## Known gaps

- Hours come from the dataset snapshot, so "open now" means open per the hours listed when the data was captured.