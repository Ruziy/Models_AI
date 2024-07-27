import tensorflow as tf
from tensorflow.keras import layers, models

class BasicBlock(tf.keras.Model):
    def __init__(self, filters):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x += residual
        return tf.nn.relu(x)

class ExchangeBlock(tf.keras.Model):
    def __init__(self, filters, downsample=False):
        super(ExchangeBlock, self).__init__()
        self.blocks = [BasicBlock(filter_num) for filter_num in filters]
        self.downsample = downsample
        if downsample:
            self.downsample_layer = layers.Conv2D(filters[-1], 3, strides=2, padding='same')

    def call(self, inputs, training=False):
        outputs = []
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1 and self.downsample:
                outputs.append(self.downsample_layer(inputs[-1]))
            else:
                outputs.append(block(inputs[i], training=training))
        return outputs

class HRNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(HRNet, self).__init__()
        self.stage1 = ExchangeBlock([16])
        self.stage2 = ExchangeBlock([16, 32], downsample=True)
        self.stage3 = ExchangeBlock([16, 32, 64], downsample=True)
        self.stage4 = ExchangeBlock([16, 32, 64, 128], downsample=True)
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = [inputs]
        x = self.stage1(x, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)

        # Слияние всех разрешений
        x_resized = [tf.image.resize(img, (28, 28)) for img in x]  
        x_concat = tf.concat(x_resized, axis=-1)

        # Применение глобального усреднения и полносвязного слоя
        x = self.global_avg_pool(x_concat)
        return self.fc(x)

# Загрузка и подготовка данных MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0  # Нормализация и добавление канала
x_test = x_test[..., tf.newaxis] / 255.0

# Инициализация и компиляция модели HRNet для задачи классификации
num_classes = 10  # Количество классов для MNIST

model = HRNet(num_classes=num_classes)
model.build(input_shape=(None, 28, 28, 1))  # Размер изображений MNIST и каналов

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))