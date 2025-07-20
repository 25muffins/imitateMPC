import fastStrafe
import tensorflow as tf
model = fastStrafe.load_model()
model.export("fastStrafeSavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("fastStrafeSavedModel")
tflite_model = converter.convert()

with open('fastStrafe.tflite', 'wb') as f:
  f.write(tflite_model)