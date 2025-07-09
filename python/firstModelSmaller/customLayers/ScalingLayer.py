#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    08-Jul-2025 20:24:07

import tensorflow as tf
import sys     # Remove this line after completing the layer definition.

class ScalingLayer(tf.keras.layers.Layer):
    # Add any additional layer hyperparameters to the constructor's
    # argument list below.
    def __init__(self, name=None):
        super(ScalingLayer, self).__init__(name=name)

    def call(self, input1):
        # Add code to implement the layer's forward pass here.
        # The input tensor format(s) are: BC
        # The output tensor format(s) are: BC
        # where B=batch, C=channels, T=time, S=spatial(in order of height, width, depth,...)

        # Remove the following 3 lines after completing the custom layer definition:
        print("Warning: load_model(): Before you can load the model, you must complete the definition of custom layer ScalingLayer in the customLayers folder.")
        print("Exiting...")
        sys.exit("See the warning message above.")

        return output1
