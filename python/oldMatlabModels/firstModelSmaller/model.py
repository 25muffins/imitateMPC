#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    08-Jul-2025 20:24:07

from tensorflow import keras
from tensorflow.keras import layers
from firstModelSmaller.customLayers.ScalingLayer import ScalingLayer

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
    fc_5 = layers.Dense(4, name="fc_5_")(relu_4)
    layer = layers.Activation('tanh')(fc_5)
    #scaling = ScalingLayer()(layer)

    model = keras.Model(inputs=[input], outputs=[layer])
    return model
