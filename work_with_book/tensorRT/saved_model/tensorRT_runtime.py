import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import time
# import onnxruntime as ort

# Оптимизация модели с TensorRT и сохранение
# def optimize_model(saved_model_dir, precision_mode='FP16', output_saved_model_dir='tensorrt_mnist_model'):
#     print(f"Optimizing model from {saved_model_dir} to {output_saved_model_dir} with precision mode {precision_mode}...")
#     params = trt.DEFAULT_TRT_CONVERSION_PARAMS
#     params = params._replace(precision_mode=precision_mode)
#     converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=params)
#     converter.convert()
#     converter.save(output_saved_model_dir=output_saved_model_dir)
#     print("Model optimized and saved.")
#     return output_saved_model_dir

# Инференс с TensorRT и замер времени
def infer_tensorrt(saved_model_dir, input_data):
    print(f"Loading optimized model from {saved_model_dir}...")
    saved_model_loaded = tf.saved_model.load(saved_model_dir)
    infer = saved_model_loaded.signatures['serving_default']
    
    input_name = list(infer.structured_input_signature[1].keys())[0]
    output_name = list(infer.structured_outputs.keys())[0]
    
    print("Running inference with TensorRT...")
    start_time = time.time()
    output = infer(tf.convert_to_tensor(input_data))
    end_time = time.time()
    
    print(f"TensorRT Inference Time: {end_time - start_time:.6f} seconds")
    return output[output_name]

# Пример входных данных
input_data = np.random.random((1, 28, 28, 1)).astype(np.float32)
print("Input data shape:", input_data.shape)

# Инференс и замер времени TensorRT
output_tensorrt = infer_tensorrt("tensorrt_mnist_model", input_data)
print("TensorRT Output:", output_tensorrt)

#docker
# docker pull nvcr.io/nvidia/tensorflow:22.06-tf2-py3 
#docker run --gpus all -it --rm -v /c/Users/Alex/Desktop/test_work_with_neyron/drafts_AI/work_with_book:/workspace/work_with_book nvcr.io/nvidia/tensorflow:22.06-tf2-py3
