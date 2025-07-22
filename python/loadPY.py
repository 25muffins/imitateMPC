import linearNetwork
import tensorflow as tf
model = linearNetwork.load_model()
model.export("linearNetworkSavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("linearNetworkSavedModel")
tflite_model = converter.convert()

with open('linearNetwork.tflite', 'wb') as f:
  f.write(tflite_model)