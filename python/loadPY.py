import v30Network
import tensorflow as tf
model = v30Network.load_model()
model.export("v30SavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("v30SavedModel")
tflite_model = converter.convert()

with open('v30.tflite', 'wb') as f:
  f.write(tflite_model)