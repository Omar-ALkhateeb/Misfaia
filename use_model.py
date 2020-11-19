from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
# import pickle
import re
import string
import pandas as pd
import numpy as np


df = pd.read_csv("filtered_ar_tweets.csv", encoding="utf-8")


def create_vectorization_layer():

    def custom_standardization(input_data):
        stripped_html = tf.strings.regex_replace(input_data, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=100000,
        output_mode='int',
        output_sequence_length=138)

    # Make a text-only dataset (without labels), then call adapt
    vectorize_layer.adapt(np.array(df['Feed'].tolist()))

    return vectorize_layer


def create_model():
    # loading only base model beacuse saving vector layer isnt implemented

    model = tf.keras.models.load_model("saved_model/my_model.h5")

    export_model = tf.keras.Sequential([
        create_vectorization_layer(),
        model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    return export_model


examples = [
    "قليل اخلاق",
    "المثيرة للجدل",
    "اخي في الله",
    "السلام عليكم",
    'ابو الشباب راعي العود ليش ماوزنه في البيت غبا ',
    "إذا تم العقل نقص الكلام"
]


model = create_model()
print(model.predict(examples))
