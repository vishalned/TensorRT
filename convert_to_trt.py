from tensorflow.python.compiler.tensorrt import trt_convert as trt

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
  precision_mode='FP16')

converter = trt.TrtGraphConverterV2(
      input_saved_model_dir='saved_models',
      conversion_params=params)
converter.convert()

saved_model_dir_trt = 'model.trt'
converter.save(saved_model_dir_trt)
