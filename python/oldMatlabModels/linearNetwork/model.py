#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    20-Jul-2025 17:35:40

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input = keras.Input(shape=(8,))
    fc_1 = layers.Dense(450, name="fc_1_")(input)
    relu_1 = layers.ReLU()(fc_1)
    fc_2 = layers.Dense(400, name="fc_2_")(relu_1)
    relu_2 = layers.ReLU()(fc_2)
    fc_3 = layers.Dense(300, name="fc_3_")(relu_2)
    relu_3 = layers.ReLU()(fc_3)
    fc_4 = layers.Dense(200, name="fc_4_")(relu_3)
    relu_4 = layers.ReLU()(fc_4)
    fc_5 = layers.Dense(100, name="fc_5_")(relu_4)
    relu_5 = layers.ReLU()(fc_5)
    fc_6 = layers.Dense(45, name="fc_6_")(relu_5)
    relu_6 = layers.ReLU()(fc_6)
    fc_7 = layers.Dense(4, name="fc_7_")(relu_6)
    layer = layers.Activation('tanh')(fc_7)

    model = keras.Model(inputs=[input], outputs=[layer])
    return model
