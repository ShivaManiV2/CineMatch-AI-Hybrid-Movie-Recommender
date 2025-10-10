import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
print(ratings['userId'].unique()[:10])
