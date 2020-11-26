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


# create and return the basic model with no vecorization
def create_base_model(X_train, X_test, y_train, y_test, max_features):

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
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=epochs)

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    history_dict.keys()

    # save and return for analysis
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    return model, acc, val_acc, loss, val_loss, epochs


# create the vectorization layer
def create_vectorization_layer(data):

    def custom_standardization(input_data):
        stripped_html = tf.strings.regex_replace(input_data, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

    sequence_length = max([len(i) for i in data if type(i) != float])

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=100000,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    vectorize_layer.adapt(np.array(data.tolist()))

    def vectorize_text(text, vectorize_layer):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text)

    return vectorize_layer, vectorize_text
