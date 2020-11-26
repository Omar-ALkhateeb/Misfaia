from flask import Flask
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
# import pickle
import re
import string
import pandas as pd
import numpy as np
from base_model import create_base_model, create_vectorization_layer
from export_model import create_export_model


df = pd.read_csv("filtered_ar_tweets.csv", encoding="utf-8")
vectorize_layer, _ = create_vectorization_layer(df['Feed'])

model = create_export_model(vectorize_layer)

# init server
app = Flask(__name__)


examples = [
    "قليل اخلاق",
    "المواضيع المثيرة للجدل",
    "اخي في الله",
    "السلام عليكم",
    'ابو الشباب راعي العود ليش ماوزنه في البيت غبا ',
    "إذا تم العقل نقص الكلام"
]


# TODO
# update curses list to a txt/csv file
# add a load handler
# ingestion engine?
# logger, security headers, cors
# auth procces?


# proto
curses = set(['damn', 'crap'])

# bad word filter
@app.route('/removeCurses/<words>')
def removeCurses(words=None):
    # lowercase our set to easily match bad words
    setWords = set(words.lower().split())
    words = words.split()
    badWords = setWords.intersection(curses)
    print(words)
    if len(badWords) > 0:
        res = [i if i not in badWords else '*'*len(i) for i in words]
        return ' '.join(res)

    return ' '.join(words)


# ml model prediction from 0 "bad" to 1 "good"
@app.route('/getPrecentage/<words>')
def getPrecentage(words=None):
    # workaround tensorflow needing array input
    return str(model.predict([words, words, words])[0][0])
    # print(model.predict([words, words, words]))


# sentence filter and cleaner
@app.route('/cleanCorpus/<words>')
def cleanCorpus(words=None):
    # print(words)
    # fix regex to keep punctuation and remove other chars
    filtered = re.sub('[^A-Za-z0-9ا-ي]+', ' ', words)
    return filtered


app.run(port=12000)
