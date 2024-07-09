import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tf2onnx
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# # Step 1: Create and save the MNIST model
# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
# test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
# tf.saved_model.save(model, 'mnist_saved_model')

# Step 2: Convert the TensorFlow model to ONNX
# saved_model_dir = 'work_with_book\\tensorRT\saved_model\mnist_saved_model'
# onnx_model_path = 'mnist_model.onnx'
# model_proto, _ = tf2onnx.convert.from_saved_model(saved_model_dir, output_path=onnx_model_path)

# Step 3: Inference with ONNX Runtime
def infer_onnxruntime(onnx_file_path, input_data):
    session = ort.InferenceSession(onnx_file_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    output = session.run([output_name], {input_name: input_data})
    end_time = time.time()

    print("ONNX Runtime Inference Time: {:.6f} seconds".format(end_time - start_time))
    return output

input_data = np.random.random((1, 28, 28, 1)).astype(np.float32)
output_onnx = infer_onnxruntime('work_with_book\\tensorRT\saved_model\mnist_model.onnx', input_data)
print("ONNX Output:", output_onnx)

# Step 4: Build and save the TensorRT engine
def infer_onnxruntime(onnx_file_path, input_data):
    session = ort.InferenceSession(onnx_file_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    output = session.run([output_name], {input_name: input_data})
    end_time = time.time()

    print("ONNX Runtime Inference Time: {:.6f} seconds".format(end_time - start_time))
    return output

# Пример входных данных
input_data = np.random.random((1, 28, 28, 1)).astype(np.float32)
output_onnx = infer_onnxruntime('work_with_book\\tensorRT\saved_model\mnist_model.onnx', input_data)
print("ONNX Output:", output_onnx)

