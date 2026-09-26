import duckdb

con = duckdb.connect()

df = con.execute("""
WITH ranked AS (
  SELECT business_id, stars, text,
         row_number() OVER (
           PARTITION BY business_id, (stars >= 4)
           ORDER BY length(text) DESC
         ) AS rn
  FROM 'data/processed/reviews.parquet'
  WHERE length(text) BETWEEN 200 AND 2000
)
SELECT r.business_id,
       b.name,
       b.categories,
       string_agg(r.text, ' ') AS blob
FROM ranked r
JOIN 'data/processed/restaurants.parquet' b USING (business_id)
WHERE r.rn <= 15
GROUP BY r.business_id, b.name, b.categories
""").df()

print(len(df))
print(df.iloc[0]['name'])
print(df.iloc[0]['blob'][:600])


from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = (df['name'] + '. ' + df['categories'] + '. ' + df['blob']).tolist()

emb = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save('data/processed/embeddings.npy', emb.astype('float32'))
df['business_id'].to_csv(
    'data/processed/business_ids.txt', index=False, header=False
)

print(emb.shape)