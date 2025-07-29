import v19Network
import tensorflow as tf
model = v19Network.load_model()
model.export("v19SavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("v19SavedModel")
tflite_model = converter.convert()

with open('v19.tflite', 'wb') as f:
  f.write(tflite_model)