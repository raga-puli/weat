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

groups.parquet (src/groups.py, built from train_reviews only)
  group_id (str): 12-char content hash of members + target
  members (list[str]): 3-4 distinct user_ids, all from the val split, sorted
  target_business_id (str): held-out restaurant every member rated >= 4 stars within the window
  window_start (timestamp): earliest member's review of the target; all members' reviews fall within 30 days of it
  size (int): number of members (3 or 4)
  split (str): "val" or "test", 70/30, deterministic from SEED + group_id

## Known gaps

- Hours come from the dataset snapshot, so "open now" means open per the hours listed when the data was captured.
- Groups are co-rating groups: strangers who liked the same place within 30 days, not real dining parties. Targets skew toward popular restaurants, and the same targets appear in both val and test groups.