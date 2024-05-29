import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Определение модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(300, activation='elu'),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.Dense(100, activation='elu'),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.Dense(10,activation = 'softmax')
])

# Компиляция модели
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=loss_fn,
              metrics=['accuracy'])

# Обучение модели
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 5
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)

for epoch in range(epochs):
    for images, labels in train_dataset:
        loss = train_step(images, labels)
    print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) ##accuracy: 0.9768 relu,without BatchNorm
print(f'Test accuracy: {test_acc}')                             ##accuracy:  0.9656 elu,BatchNorm