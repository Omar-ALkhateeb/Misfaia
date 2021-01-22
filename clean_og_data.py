import pandas as pd
# used to load and filter the original data

df = pd.read_excel("AJGT.xlsx", engine='openpyxl')

# df.head()


df.Sentiment = [1 if i == 'Positive' else 0 for i in df.Sentiment.values]

df.to_csv('filtered_ar_tweets.csv', index=False)
