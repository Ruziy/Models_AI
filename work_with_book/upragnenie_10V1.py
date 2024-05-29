import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
#Загрузка данных
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

#Параметры для MNIST
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.compat.v1.placeholder(tf.int32, shape=(None), name="y")

#Реализация слоя
def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        #Получение признаков
        n_inputs = int(X.get_shape()[1])
        #Стандартное отклонение
        stddev = 2 / np.sqrt(n_inputs)
        #Инициализация усеченных весов
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)                
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

#Модель нейросети
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X,n_hidden1,name="hidden1",activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1,n_hidden2,name="hidden2",activation=tf.nn.relu)
    logits = neuron_layer(hidden2,n_inputs,name="outputs")
    

#Функция потерь 
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

#Функция и скорость обучения
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#Функция точности 
with tf.name_scope("eval"):
    correct = tf.keras.backend.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#Рандомайзер для разбиения по пакетам
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

#Запуск основных переменных
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

#Параметры обучения
n_epochs = 40
batch_size = 50
max_val = 0
#Запуск модели
with tf.compat.v1.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
        max_val = max(acc_val,max_val)
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
print("Max accuracy: ", max_val)    ##0.9784

    # save_path = saver.save(sess, "./my_model_final.ckpt")