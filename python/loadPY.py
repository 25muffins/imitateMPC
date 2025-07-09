import firstModelSmaller
import tensorflow as tf
model = firstModelSmaller.load_model()
model.export("savedModelSmaller")
converter = tf.lite.TFLiteConverter.from_saved_model("savedModelSmaller")
tflite_model = converter.convert()

with open('modelSmaller.tflite', 'wb') as f:
  f.write(tflite_model)