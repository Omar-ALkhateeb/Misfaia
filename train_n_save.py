#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from sklearn import model_selection
import re
import pandas as pd
from base_model import create_base_model, create_vectorization_layer
from export_model import create_export_model
import pickle


df = pd.read_csv("filtered_ar_tweets.csv", encoding="utf-8")

df = df.dropna()
df = df.sample(frac=1)

print(df.head())


vectorize_layer, vectorize_text = create_vectorization_layer(df['Feed'])

print("Review", df['Feed'][0])
print("Label", df['Sentiment'][0])
print("Vectorized review", vectorize_text(df['Feed'][:1], vectorize_layer))


print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


X_train, X_test, y_train, y_test = model_selection.train_test_split(df['Feed'],
                                                                    df['Sentiment'], test_size=0.20)

# print(vectorize_text(X_train))
train_ds = vectorize_text(X_train, vectorize_layer)
val_ds = vectorize_text(X_test, vectorize_layer)


train_ds
model, acc, val_acc, loss, val_loss, epochs = create_base_model(
    train_ds, val_ds, y_train, y_test)
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


export_model = create_export_model(vectorize_layer)


# print(X_test[0])

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


# # saving the serialize seperately bc we cant call model.save on it
# f = open('vectotizer.bin', 'wb')
# pickle.dump(vectorize_layer, f)

# # 3.3. Close file
# f.close()

# saving only base model beacuse saving vector layer isnt implemented
model.save('saved_model/my_model.h5',
           overwrite=True, include_optimizer=True)
