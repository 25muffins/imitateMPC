import v16Network
import tensorflow as tf
model = v16Network.load_model()
model.export("v16SavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("v16SavedModel")
tflite_model = converter.convert()

with open('v16.tflite', 'wb') as f:
  f.write(tflite_model)