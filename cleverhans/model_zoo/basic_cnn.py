"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

from cleverhans import initializers
from cleverhans.model import Model


class ModelBasicCNN(Model):
  def __init__(self, scope, nb_classes, nb_filters, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters

    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [128, 28, 28, 1]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                kernel_initializer=initializers.HeReLuNormalInitializer)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = my_conv(x, self.nb_filters, 8, strides=2, padding='same')
      y = my_conv(y, 2 * self.nb_filters, 6, strides=2, padding='valid')
      y = my_conv(y, 2 * self.nb_filters, 5, strides=1, padding='valid')
      logits = tf.layers.dense(tf.layers.flatten(y), self.nb_classes,
                               kernel_initializer=initializers.HeReLuNormalInitializer)
      return {self.O_LOGITS: logits, self.O_PROBS: tf.nn.softmax(logits=logits)}



class ModelHWCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters
        self.fprop(tf.placeholder(tf.float32, [128, 64, 64, 1]))
        self.params = self.get_params()


    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=initializers.HeReLuNormalInitializer)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = my_conv(x, self.nb_filters, 8, strides=2, padding='same')
            y = my_conv(y, 2 * self.nb_filters, 6, strides=2, padding='valid')
            y = my_conv(y, 2 * self.nb_filters, 5, strides=1, padding='valid')
            logits = tf.layers.dense(
                tf.layers.flatten(y), self.nb_classes,
                kernel_initializer=initializers.HeReLuNormalInitializer)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class ModelOCRCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        self.train_phase = kwargs['train_phase']
        self.keep_prob = kwargs['keep_prob']
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters
        self.fprop(tf.placeholder(tf.float32, [10, 32, 256, 1]))
        self.params = self.get_params()

    ## http://stackoverflow.com/a/34634291/2267819
    def batch_norm(self, x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
        with tf.variable_scope(scope):
            # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
            # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        return normed


    def fprop(self, x, **kwargs):
        train_phase = self.train_phase
        keep_prob = self.keep_prob
        del kwargs
        w_alpha = 0.01
        b_alpha = 0.1
        x = tf.reshape(x, shape=[-1, 32, 256, 1])
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # 4 conv layer
            w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
            b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
            conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
            conv1 = self.batch_norm(conv1, tf.constant(0.0, shape=[32]),
                               tf.random_normal(shape=[32], mean=1.0, stddev=0.02),
                               train_phase, scope='bn_1')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, keep_prob)

            w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
            b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
            conv2 = self.batch_norm(conv2, tf.constant(0.0, shape=[64]),
                               tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                               train_phase, scope='bn_2')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, keep_prob)

            w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
            b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
            conv3 = self.batch_norm(conv3, tf.constant(0.0, shape=[64]),
                               tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                               train_phase, scope='bn_3')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.dropout(conv3, keep_prob)

            w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
            b_c4 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
            conv4 = self.batch_norm(conv4, tf.constant(0.0, shape=[64]),
                               tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                               train_phase, scope='bn_4')
            conv4 = tf.nn.relu(conv4)
            conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv4 = tf.nn.dropout(conv4, keep_prob)

            # Fully connected layer
            w_d = tf.Variable(w_alpha * tf.random_normal([2 * 16 * 64, 1024]))
            b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
            dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
            dense = tf.nn.dropout(dense, keep_prob)

            w_out = tf.Variable(w_alpha * tf.random_normal([1024, 50]))
            b_out = tf.Variable(b_alpha * tf.random_normal([50]))
            out = tf.add(tf.matmul(dense, w_out), b_out)

        return {self.O_LOGITS: out,
                self.O_PROBS: tf.nn.softmax(logits=out)}

from tensorflow.python.platform import gfile
class ModelChiOCR(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters
        # self.fprop(tf.placeholder(tf.float32, [128, 64, 64, 1]))
        # self.params = self.get_params()
        self.sess = None
        self.load_model()

    def load_model(self):
        pb_file_path = "/home/chenlu/research/clevertest/trans_model/densenet.pb"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

    def get_inputs(self):
        return sess.graph.get_tensor_by_name('the_input_4:0')

    def get_logits(self, x):
        # TODO: x connect to tensor('input_1:0')
        return sess.graph.get_tensor_by_name('output_1:0')

    def get_probs(self, x):
        return sess.graph.get_tensor_by_name('output_1:0')



