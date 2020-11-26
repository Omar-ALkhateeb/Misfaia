import os
import pandas as pd
import re

# used to load and filter the original data
# really bad code i know.

data = []
labels = []
allLines = []
pos_path = 'arabic_tweets/pos/'
pos_fileList = os.listdir(pos_path)
for f in pos_fileList:
    txt = open(os.path.join(pos_path + f), 'r', encoding='utf-8')
    allLines.append(txt.read())
    txt.close()
data.extend([1 for _ in pos_fileList])

neg_path = 'arabic_tweets/neg/'
neg_fileList = os.listdir(neg_path)
for f in neg_fileList:
    txt = open(os.path.join(neg_path + f), 'r', encoding='utf-8')
    allLines.append(txt.read())
    txt.close()
data.extend([0 for _ in neg_fileList])
print(allLines)


print(allLines[-7:])
print(labels[-7:])
df = pd.DataFrame()
data = [re.sub('[^A-Za-z0-9ا-ي]+', ' ', i) for i in data]
df['Feed'] = data
df['Sentiment'] = labels


labels, data = [], []

for i in df['Feed']:
    x = re.sub('[^A-Za-z0-9ا-ي]+', ' ', i)
    data.append(x)

# labels = [1 if i!='Negative' else 0 for i in df['Sentiment']]
data[:40]
df = pd.DataFrame()
df['Feed'] = data
df['Sentiment'] = labels


print(df['Feed'].count())
# deleting floats in Feed col (idk where they came from)
needs_deletion = [i for i in range(
    df['Feed'].count()) if type(df['Feed'][i]) == float]
for i in needs_deletion:
    # print('deleted')
    df = df.drop(df.index[i])
print(df['Feed'].count())

df.to_csv('filtered_ar_tweets.csv', index=True)
