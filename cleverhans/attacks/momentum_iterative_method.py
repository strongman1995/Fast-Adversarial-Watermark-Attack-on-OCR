"""The MomentumIterativeMethod attack.
"""

import warnings
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from cleverhans.attacks.attack import Attack
from cleverhans.attacks.fast_gradient_method import optimize_linear
from cleverhans.attacks.fast_gradient_method import optimize_linear_pos
from cleverhans.compat import reduce_sum, reduce_mean, softmax_cross_entropy_with_logits
from cleverhans import utils_tf

tf.logging.set_verbosity(tf.logging.DEBUG)


class MomentumIterativeMethod(Attack):
    """
    The Momentum Iterative Method (Dong et al. 2017). This method won
    the first places in NIPS 2017 Non-targeted Adversarial Attacks and
    Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf

    :param model: cleverhans.model.Model
    :param sess: optional tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a MomentumIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(MomentumIterativeMethod, self).__init__(model, sess, dtypestr,
                                                      **kwargs)
        self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min', 'clip_max')
        self.structural_kwargs = ['ord', 'nb_iter', 'decay_factor', 'sanity_checks']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param kwargs: Keyword arguments. See `parse_params` for documentation.
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        asserts = []

        # If a data range was specified, check that the input was in that range
        # if self.clip_min is not None:
        #     asserts.append(utils_tf.assert_greater_equal(x, tf.cast(self.clip_min, x.dtype)))

        # if self.clip_max is not None:
        #     asserts.append(utils_tf.assert_less_equal(x, tf.cast(self.clip_max, x.dtype)))

        # Initialize loop variables
        momentum = tf.zeros_like(x)
        adv_x = x + self.init_mask * self.mask# initial noise

        # Fix labels to the first model predictions for loss computation
        y = self.y_target
        loss_type = "softmax"
        if loss_type == "softmax":
            y = y / tf.reduce_sum(y, -1, keepdims=True)
        targeted = (self.y_target is not None)

        def cond(i, _, __):
            """Iterate until number of iterations completed"""
            # i = tf.Print(i, [i], message="in cond")
            return tf.less(i, self.nb_iter)

        def body(i, ax, m):
            """Do a momentum step"""
            if loss_type == 'softmax':
                logits = self.model.get_logits(ax)
                early_stop = False
                if early_stop:
                    # i = tf.cond(tf.less(loss, early_stop_loss_threshold), lambda: self.nb_iter, lambda: i)
                    max_y = tf.argmax(y, axis=-1, name='max_y')
                    max_logits = tf.argmax(logits, axis=-1, name='max_logits')
                    eq = tf.equal(max_y, max_logits)
                    eq = tf.cast(eq, dtype=tf.float32)
                    cnt_eq = tf.reduce_sum(1 - eq)
                    # len_txt = max_y.get_shape().as_list()[1]
                    tot_eq = tf.equal(cnt_eq, 0)
                    i = tf.cond(tot_eq, lambda: self.nb_iter, lambda: i)
                loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
                loss = tf.reduce_mean(loss, name='softmax_loss')
            elif loss_type == "ctc":
                time_major_logits, output_seq_len = self.model.get_logits(ax)
                ctc_loss = tf.nn.ctc_loss(labels=y,
                                          inputs=time_major_logits,
                                          sequence_length=output_seq_len,
                                          time_major=True,
                                          ctc_merge_repeated=True,
                                          ignore_longer_outputs_than_inputs=True)
                loss = tf.reduce_mean(ctc_loss, name='ctc_loss')

            if targeted:
                loss = -loss

            # Define gradient of loss wrt input
            grad, = tf.gradients(loss, ax)

            # Normalize current gradient and add it to the accumulated gradient
            red_ind = list(range(1, len(grad.get_shape())))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.maximum(avoid_zero_div, tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True))
            m = self.decay_factor * m + grad

            # optimal_perturbation = optimize_linear(m, self.eps_iter, self.ord)
            optimal_perturbation = optimize_linear_pos(m, self.eps_iter, self.ord, self.pert_type)
            optimal_perturbation = tf.multiply(optimal_perturbation, self.mask, name="op_multiply")
            if self.ord == 1:
                raise NotImplementedError("This attack hasn't been tested for ord=1. It's not clear that FGM makes a good inner loop step "
                                          "for iterative optimization since it updates just one coordinate at a time.")

            # Update and clip adversarial example in current iteration
            ax = ax + optimal_perturbation
            ax = x + utils_tf.clip_eta(ax - x, self.ord, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                ax = utils_tf.clip_by_value(ax, self.clip_min, self.clip_max)

            ax = tf.stop_gradient(ax)
            return i + 1, ax, m

        _, adv_x, _ = tf.while_loop(cond, body, (tf.zeros([]), adv_x, momentum), back_prop=True)

        if self.sanity_checks:
            with tf.control_dependencies(asserts):
                adv_x = tf.identity(adv_x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.06, nb_iter=10, y=None, ord=np.inf, decay_factor=1.0,
                     clip_min=None, clip_max=None, y_target=None, sanity_checks=False, mask=None, init_mask=0, pert_type='all', **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) maximum distortion of adversarial example compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the true labels.
        :param y_target: (optional) A tensor with the labels to target. Leave y_target=None if y is also set.
                          Labels should be one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.decay_factor = decay_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sanity_checks = sanity_checks
        self.mask = mask
        self.init_mask = init_mask
        self.pert_type = pert_type


        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after 2019-04-26.")

        return True
