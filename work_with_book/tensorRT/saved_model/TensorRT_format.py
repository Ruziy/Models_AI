import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Путь к сохраненной модели TensorFlow (SavedModel)
saved_model_dir = 'mnist_saved_model'

# Оптимизация с помощью tf.experimental.tensorrt.Converter
params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params = params._replace(precision_mode='FP16')  # Например, FP16 режим
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=params)
converter.convert()

# Сохранение оптимизированной модели TensorRT
converter.save(output_saved_model_dir='tensorrt_mnist_model')
