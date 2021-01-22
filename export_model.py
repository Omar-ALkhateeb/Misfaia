from base_model import create_vectorization_layer
from tensorflow.keras.models import load_model

import tensorflow as tf
import pandas as pd


def create_export_model(vectorize_layer):
    # loading only base model beacuse saving vector layer isnt implemented

    model = load_model("saved_model/my_model.h5")

    export_model = tf.keras.models.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    return export_model
