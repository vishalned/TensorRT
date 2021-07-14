import tensorflow as tf
import numpy as np

params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="./saved_models", conversion_params=params)
converter.convert()

def my_input_fn():
  input_sizes = [160, 160]
  inp1 = np.random.normal(size=(1, *input_sizes, 80, 1)).astype(np.float32)
  yield [inp1]

converter.build(input_fn = my_input_fn)
converter.save('./model_tf.trt')
