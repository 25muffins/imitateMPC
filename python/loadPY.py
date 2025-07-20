import fastStrafe2
import tensorflow as tf
model = fastStrafe2.load_model()
model.export("fastStrafeSavedModel2")
converter = tf.lite.TFLiteConverter.from_saved_model("fastStrafeSavedModel2")
tflite_model = converter.convert()

with open('fastStrafe2.tflite', 'wb') as f:
  f.write(tflite_model)