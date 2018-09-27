import tensorflow as tf
import numpy as np

with tf.device('/device:GPU:0'):
    mu = tf.Variable([1000000, 1], dtype=tf.float32)
    sig = tf.Variable([1000000, 1], dtype=tf.float32)

    tf.distributions.Normal(mu[0], sig[0])

    tf.map_fn(tf.distributions.Normal, (mu, sig), dtype=(tf.float32,)*1000000)
    
    normals = [tf.distributions.Normal(mu[i], sig[i]**2) for i in range(1000000)]
