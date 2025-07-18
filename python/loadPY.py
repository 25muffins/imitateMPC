import relativeDistAngleModel
import tensorflow as tf
model = relativeDistAngleModel.load_model()
model.export("relativeDASavedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("relativeDASavedModel")
tflite_model = converter.convert()

with open('relativeDistAngle.tflite', 'wb') as f:
  f.write(tflite_model)