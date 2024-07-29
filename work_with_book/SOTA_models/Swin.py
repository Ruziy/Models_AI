import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model

# Загрузка и подготовка данных MNIST
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class WindowAttention(Layer):
    def __init__(self, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=window_size)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
        
    def call(self, inputs):
        x = self.layernorm(inputs)
        attn_output = self.attention(x, x)
        return self.dropout(attn_output) + inputs

class SwinBlock(Layer):
    def __init__(self, num_heads, window_size, shift_size, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = WindowAttention(num_heads, window_size)
        self.mlp = tf.keras.Sequential([
            Dense(mlp_dim, activation='relu'),
            Dense(window_size * window_size),
        ])
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
        
    def call(self, inputs):
        H, W, C = inputs.shape[1:]
        if self.shift_size > 0:
            shifted_inputs = tf.roll(inputs, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_inputs = inputs
        
        x = self.attn(shifted_inputs)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        
        x = self.layernorm(x)
        x = self.mlp(x)
        return self.dropout(x) + inputs

# Определение входных данных
inputs = tf.keras.Input(shape=(28, 28, 1))

# Swin Transformer блоки
x = SwinBlock(num_heads=4, window_size=7, shift_size=0, mlp_dim=128)(inputs)
x = SwinBlock(num_heads=4, window_size=7, shift_size=3, mlp_dim=128)(x)

# Глобальный пуллинг и классификация
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

# Создание и компиляция модели
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
