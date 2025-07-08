#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    08-Jul-2025 10:55:23

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from firstModel.customLayers.ScalingLayer import ScalingLayer

def create_model():
    input = keras.Input(shape=(7,))
    fc_1 = layers.Dense(45, name="fc_1_")(input)
    relu_1 = layers.ReLU()(fc_1)
    fc_2 = layers.Dense(45, name="fc_2_")(relu_1)
    relu_2 = layers.ReLU()(fc_2)
    fc_3 = layers.Dense(45, name="fc_3_")(relu_2)
    relu_3 = layers.ReLU()(fc_3)
    fc_4 = layers.Dense(45, name="fc_4_")(relu_3)
    relu_4 = layers.ReLU()(fc_4)
    fc_5 = layers.Dense(45, name="fc_5_")(relu_4)
    relu_5 = layers.ReLU()(fc_5)
    fc_6 = layers.Dense(45, name="fc_6_")(relu_5)
    relu_6 = layers.ReLU()(fc_6)
    fc_7 = layers.Dense(45, name="fc_7_")(relu_6)
    relu_7 = layers.ReLU()(fc_7)
    fc_8 = layers.Dense(45, name="fc_8_")(relu_7)
    relu_8 = layers.ReLU()(fc_8)
    fc_9 = layers.Dense(45, name="fc_9_")(relu_8)
    relu_9 = layers.ReLU()(fc_9)
    fc_10 = layers.Dense(4, name="fc_10_")(relu_9)
    layer = layers.Activation('tanh')(fc_10)
    #scaling = ScalingLayer()(layer)

    model = keras.Model(inputs=[input], outputs=[layer])
    return model
