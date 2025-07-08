import firstModel
import tensorflow as tf
model = firstModel.load_model()
model.export("savedModel")
converter = tf.lite.TFLiteConverter.from_saved_model("savedModel")
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)