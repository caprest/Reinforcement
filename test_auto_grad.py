import tensorflow as tf
import numpy as np

x = tf.Variable(np.random.random_sample(), dtype=tf.float32)
y = tf.Variable(np.random.random_sample(), dtype=tf.float32)
def cons(x):
    return tf.constant(x, dtype=tf.float32)

f = tf.pow(x, cons(2)) + cons(2) * x * y + cons(3) * tf.pow(y, cons(2)) + cons(4) * x + cons(5) * y + cons(6)

gradient = tf.gradients(f,[x,y])


sess = tf.Session()
sess.run(global_variables_initializer())
sess.run(hessian)