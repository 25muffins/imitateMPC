#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    19-Jul-2025 14:00:00

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input = keras.Input(shape=(8,))
    fc_1 = layers.Dense(450, name="fc_1_")(input)
    relu_1 = layers.ReLU()(fc_1)
    dropout = layers.Dropout(0.200000)(relu_1)
    fc_2 = layers.Dense(450, name="fc_2_")(dropout)
    relu_2 = layers.ReLU()(fc_2)
    fc_3 = layers.Dense(450, name="fc_3_")(relu_2)
    relu_3 = layers.ReLU()(fc_3)
    fc_4 = layers.Dense(450, name="fc_4_")(relu_3)
    relu_4 = layers.ReLU()(fc_4)
    fc_5 = layers.Dense(4, name="fc_5_")(relu_4)
    layer = layers.Activation('tanh')(fc_5)

    model = keras.Model(inputs=[input], outputs=[layer])
    return model
