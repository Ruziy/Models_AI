import tensorflow as tf

# Загрузка модели из файла .keras
model = tf.keras.models.load_model('work_with_book/tensorRT/saved_model/mnist_saved_model.keras')

# Сохранение модели в формате SavedModel
tf.saved_model.save(model, 'work_with_book\\tensorRT\saved_model\saved_mnist')

#После в консоли python -m tf2onnx.convert --saved-model mnist_saved_model --output mnist_model.onnx для перевода в ONNX