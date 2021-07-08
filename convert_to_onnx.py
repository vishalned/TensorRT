import keras
import tensorflow as tf
from keras2onnx import convert_keras
import tensorflow as tf
import onnx
import tf2onnx.convert

model = tf.keras.models.load_model('./model.hdf5', compile = False)

onnx_model, _ = tf2onnx.convert.from_keras(model, opset = 12)

onnx.save(onnx_model, './model.onnx')

#def keras_to_onnx(model, output_filename):
#   onnx = convert_keras(model, output_filename)
#   with open(output_filename, "wb") as f:
#       f.write(onnx.SerializeToString())

#model = keras.models.load_model('./model.hdf5', compile = False)
#keras_to_onnx(model, './model.onnx') 
