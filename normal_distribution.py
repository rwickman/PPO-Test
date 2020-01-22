# Based heavily on OpenAI DiagGaussianPd https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
import numpy as np
import tensorflow as tf

class NormalDistribution:
    def __init__(self, num_actions=6, mean=None):
        self.mean = mean
        self.logstd = tf.Variable([[0.6] * num_actions])
        self.std = tf.exp(self.logstd)


    def logp(self, x):
        return - self.neglogp(x)

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)
    
    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
        