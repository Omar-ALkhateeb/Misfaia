#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
# import os
import re
import pandas as pd

from tensorflow.keras.layers import Flatten, GlobalAveragePooling1D, Dense, Embedding, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import string
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tf.get_logger().setLevel('ERROR')


# used to load and filter the original data

# data = []
# labels = []
# pos_path = 'arabic_tweets/pos/'
# pos_fileList = os.listdir(pos_path)
# for f in pos_fileList:
#     txt = open(os.path.join(pos_path+ f), 'r', encoding='utf-8')
#     allLines.append(txt.read())
#     txt.close()
# data.extend([1 for _ in pos_fileList])

# neg_path = 'arabic_tweets/neg/'
# neg_fileList = os.listdir(neg_path)
# for f in neg_fileList:
#     txt = open(os.path.join(neg_path+ f), 'r', encoding='utf-8')
#     allLines.append(txt.read())
#     txt.close()
# data.extend([0 for _ in neg_fileList])
# print(allLines)


# print(allLines[-7:])
# print(labels[-7:])
# df = pd.DataFrame()
# data = [re.sub('[^A-Za-z0-9ا-ي]+', ' ', i) for i in data]
# df['Feed'] = data
# df['Sentiment'] = labels


# labels, data = [], []

# for i in df['Feed']:
#     x = re.sub('[^A-Za-z0-9ا-ي]+', ' ', i)
#     data.append(x)

# # labels = [1 if i!='Negative' else 0 for i in df['Sentiment']]
# data[:40]
# df = pd.DataFrame()
# df['Feed'] = data
# df['Sentiment'] = labels

# df.to_csv('filtered_ar_tweets.csv', index=True)


df = pd.read_csv("filtered_ar_tweets.csv", encoding="utf-8")

print(df.head())


# print(df['Feed'].count())
# # deleting floats in Feed (idk where they cane from)
# needs_deletion = [i for i in range(
#     df['Feed'].count()) if type(df['Feed'][i]) == float]
# for i in needs_deletion:
#     # print('deleted')
#     df = df.drop(df.index[i])
# print(df['Feed'].count())


def custom_standardization(input_data):
    #   lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 100000
# sequence_length = 250
sequence_length = max([len(i) for i in df['Feed'] if type(i) != float])
print('seq len', sequence_length)


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# Make a text-only dataset (without labels), then call adapt
vectorize_layer.adapt(np.array(df['Feed'].tolist()))


def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)


print("Review", df['Feed'][0])
print("Label", df['Sentiment'][0])
print("Vectorized review", vectorize_text(df['Feed'][:1]))


print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


X_train, X_test, y_train, y_test = model_selection.train_test_split(df['Feed'],
                                                                    df['Sentiment'], test_size=0.30)

# print(vectorize_text(X_train))
train_ds = vectorize_text(X_train)
val_ds = vectorize_text(X_test)


train_ds


embedding_dim = 16

model = tf.keras.Sequential([
    Embedding(max_features + 1, embedding_dim),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(1)])

model.summary()


model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


epochs = 20
history = model.fit(train_ds, y_train, validation_data=(
    val_ds, y_test), epochs=epochs)


loss, accuracy = model.evaluate(val_ds, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


history_dict = history.history
history_dict.keys()


acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    tf.keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(X_test, y_test)
print(accuracy)


examples = [
    "قليل اخلاق",
    "المثيرة للجدل",
    "اخي في الله",
    "السلام عليكم",
    'ابو الشباب راعي العود ليش ماوزنه في البيت غبا ',
    "إذا تم العقل نقص الكلام"
]

print(export_model.predict(examples))


# saving only base model beacuse saving vector layer isnt implemented
model.save('saved_model/my_model.h5',
           overwrite=True, include_optimizer=True)
