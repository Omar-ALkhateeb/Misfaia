import tensorflow as tf


def create_export_model(vectorize_layer, model):
    export_model = tf.keras.models.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    return export_model
