import MPCPathGenNetwork
import tensorflow as tf
model = MPCPathGenNetwork.load_model()
model.export("MPCPathGenSavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("MPCPathGenSavedModel")
tflite_model = converter.convert()

with open('MPCPathGen.tflite', 'wb') as f:
  f.write(tflite_model)