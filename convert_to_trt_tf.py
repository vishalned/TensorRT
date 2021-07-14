import tensorflow as tf

params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="./saved_models", conversion_params=params)
converter.convert()
converter.save('./model_tf.trt')
