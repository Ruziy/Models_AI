import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Параметры
img_height = 28
img_width = 28
num_channels = 1
num_classes = 10
patch_size = 4  # Размер патча 4x4
num_patches = (img_height // patch_size) * (img_width // patch_size)
embed_dim = 64
num_heads = 4
ff_dim = 128
num_transformer_blocks = 4
dropout_rate = 0.1

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, img_height, img_width, num_channels).astype("float32") / 255.0
x_test = x_test.reshape(-1, img_height, img_width, num_channels).astype("float32") / 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Простая реализация блока Transformer
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Простая реализация патч эмбеддинга
class PatchEmbedding(Layer):
    def __init__(self, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = Dense(embed_dim)

    def call(self, patch):
        return self.projection(patch)

# Основная модель DaViT
class DaViTModel(Model):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_transformer_blocks):
        super(DaViTModel, self).__init__()
        self.patch_embedding = PatchEmbedding(num_patches, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        self.flatten = Flatten()
        self.dense = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.flatten(x)
        return self.dense(x)

# Создание и компиляция модели
model = DaViTModel(num_patches, embed_dim, num_heads, ff_dim, num_transformer_blocks)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Обучение модели
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
