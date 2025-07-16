from oldMatlabModels import secondModelNine
import tensorflow as tf
model = secondModelNine.load_model()
model.export("savedSecondModelNine")
converter = tf.lite.TFLiteConverter.from_saved_model("savedSecondModelNine")
tflite_model = converter.convert()

with open('secondModel.tflite', 'wb') as f:
  f.write(tflite_model)