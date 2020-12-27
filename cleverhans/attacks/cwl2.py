"""The CarliniWagnerL2 attack
"""
# pylint: disable=missing-docstring
import logging

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.model import CallableModelWrapper, Model, wrapper_warning_logits
from cleverhans import utils

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.carlini_wagner_l2")
_logger.setLevel(logging.DEBUG)


class CarliniWagnerL2(Attack):
    """
    This attack was originally proposed by Carlini and Wagner.
    It is an iterative attack that finds adversarial examples on many defenses that are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and a specially-chosen loss function
    to find adversarial examples with lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model: cleverhans.model.Model
    :param sess: tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            wrapper_warning_logits()
            model = CallableModelWrapper(model, 'logits')

        super(CarliniWagnerL2, self).__init__(model, sess, dtypestr, **kwargs)

        self.feedable_kwargs = ('y', 'y_target')

        self.structural_kwargs = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_steps',
                                  'max_iterations', 'abort_early', 'initial_const', 'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given input.
        Generate uses tf.py_func in order to operate over tensors.

        :param x: A tensor with the inputs.
        :param kwargs: See `parse_params`
        """
        assert self.sess is not None, 'Cannot use `generate` when no `sess` was provided'
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        print("nb_classes", nb_classes)
        print("labels", labels)

        attack = CWL2(self.sess, self.model, self.batch_size, self.confidence, 'y_target' in kwargs, self.learning_rate,
                      self.binary_search_steps, self.max_iterations, self.abort_early, self.initial_const, self.clip_min,
                      self.clip_max, nb_classes, x.get_shape().as_list()[1:], self.wm)

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

        wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)
        wrap.set_shape(x.get_shape())

        return wrap

    def parse_params(self, y=None, y_target=None, batch_size=1, confidence=0, learning_rate=5e-3, binary_search_steps=5,
                     max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=0, clip_max=1, wm=None):
        """
        :param y: (optional) A tensor with the true labels for an untargeted attack.
                             If None (and y_target is None) then use the original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces examples with larger l2 distortion,
                           but more strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
        :param binary_search_steps: The number of times we perform binary search to find the optimal tradeoff-constant
                                    between norm of the purturbation and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this to a larger value will produce lower distortion
                               results. Using only a few iterations requires a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent is unable to make progress (i.e., gets stuck in a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the relative importance of size of the perturbation
                              and confidence of classification. If binary_search_steps is large, the initial constant is not important.
                              A smaller value of this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # ignore the y and y_target argument
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.wm = wm


def ZERO():
    return np.asarray(0., dtype=np_dtype)


class CWL2(object):
    def __init__(self, sess, model, batch_size, confidence, targeted, learning_rate, binary_search_steps, max_iterations,
                 abort_early, initial_const, clip_min, clip_max, num_labels, shape, wm):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples with larger l2 distortion, but more strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial examples produced. If set to False, they will be misclassified in any wrong class.
                         If set to True, they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
        :param binary_search_steps: The number of times we perform binary search to find the optimal tradeoff-constant
                                    between norm of the purturbation and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this to a larger value will produce lower distortion results.
                               Using only a few iterations requires a larger learning rate, and will produce larger distortion results.
        :param abort_early: If true, allows early aborts if gradient descent is unable to make progress (i.e., gets stuck in a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the relative importance of size of the pururbation and confidence of classification.
                              If binary_search_steps is large, the initial constant is not important. A smaller value of this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.wm = wm

        self.repeat = binary_search_steps >= 10  # boolean

        self.shape = shape = tuple([batch_size] + list(shape))  # (batch_size, height, width, channel)

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))  # (batch_size, height, width, channel)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')  # (batch_size, height, width, channel)

        # the resulting instance, tanh'd to keep bounded from clip_min to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2 * (clip_max - clip_min) + clip_min  # (batch_size, height, width, channel)

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)  # (batch_size, len_text, num_labels)

        # watermark region
        self.watermark = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='watermark')  # (batch_size, height, width, channel)

        label_shape = self.output.get_shape().as_list()[1:]
        # self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
        self.tlab = tf.Variable(np.zeros(tuple([batch_size] + label_shape)), dtype=tf_dtype, name='tlab')  # (batch_size, len_text, num_labels)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf_dtype, name='const')  # (batch_size, )

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')  # (batch_size, height, width, channel)
        # self.assign_tlab = tf.placeholder(tf_dtype, (batch_size, num_labels), name='assign_tlab')
        self.assign_tlab = tf.placeholder(tf_dtype, tuple([batch_size] + self.output.get_shape().as_list()[1:]), name='assign_tlab')  # (batch_size, len_text, num_labels)
        self.assign_const = tf.placeholder(tf_dtype, [batch_size], name='assign_const')  # (batch_size, )
        self.assign_watermark = tf.placeholder(tf_dtype, shape, name='assign_watermark')  # (batch_size, height, width, channel)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / 2 * (clip_max - clip_min) + clip_min  # (batch_size, height, width, channel)
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.other), axis=list(range(1, len(shape))))  # (batch_size, )

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_max(self.tlab * self.output, axis=-1)  # (batch_size, len_text)
        other = tf.reduce_max((1 - self.tlab) * self.output - self.tlab * 10000, axis=-1)  # (batch_size, len_text)
        # other = tf.reduce_max((1 - self.tlab) * self.output, axis=-1)  # (batch_size, len_text)
        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)  # (batch_size, len_text)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss1 = tf.reduce_sum(self.const * loss1)  # (1, )
        self.loss2 = tf.reduce_sum(self.l2dist)  # (1, )
        self.loss3 = tf.reduce_sum(self.watermark * tf.cast(tf.not_equal(modifier, tf.zeros(shape, dtype=tf_dtype)), dtype=tf_dtype))
        # self.loss = self.loss1 + self.loss2  # (1, )
        # self.loss = tf.reduce_sum(tf.reduce_max(real, axis=-1)) # (1, )
        self.loss = self.loss1 + self.loss2 #  + self.loss3

        # Setup the adam optimizer and
        # keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        self.train = self.optimizer.minimize(self.loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.watermark.assign(self.assign_watermark))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(f"Running CWL2 attack on instance {i} of {len(imgs)}")
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """

        # char_list = []
        # pred_text = pred.argmax(axis=2)[0]
        # for i in range(len(pred_text)):
        #   if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        #     char_list.append(characters[pred_text[i]])
        def decode(x):
            char_list = []
            for i in range(len(x)):
                if x[i] != 5989 and ((not (i > 0 and x[i] == x[i - 1])) or (i > 1 and x[i] == x[i - 2])):
                    char_list.append(x[i])
            return np.array(char_list)

        def compare(x, y): # compare(sc, lab)
            # print(x.shape, y.shape)
            x = decode(x)
            y = decode(y)
            # print("x:", x, "y:", y)
            if self.TARGETED:
                # print(f"x: {x}, y: {y}")
                return np.sum(1 - (x == y)) == 0
            else:
                return np.sum(1 - (x == y)) != 0

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)  # (full_size, height, width, channel)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)  # (full_size, height, width, channel)
        imgs = np.clip(imgs, 0, 1)  # # (full_size, height, width, channel)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1  # # (full_size, height, width, channel)
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)  # (full_size, height, width, channel)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)  # (batch_size, )
        CONST = np.ones(batch_size) * self.initial_const  # (batch_size, )
        upper_bound = np.ones(batch_size) * 1e10  # (batch_size, )

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size  # (batch_size, )
        # o_bestscore = [-1]  * batch_size  # (batch_size, )
        o_bestscore = np.ones([batch_size] + list(labs.shape[1:-1])) * (-1)  # (batch_size, len_text)
        o_bestattack = np.copy(oimgs)  # (full_size, height, width, channel)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]  # (batch_size, height, width, channel)
            batchlab = labs[:batch_size]  # (batch_size, len_text, num_labels)

            bestl2 = [1e10] * batch_size  # (batch_size, )
            # bestscore = [-1] * batch_size  # (batch_size, )
            bestscore = np.ones([batch_size] + list(labs.shape[1:-1])) * (-1)  # (batch_size, len_text)
            _logger.debug(f"  Binary search step {outer_step} of {self.BINARY_SEARCH_STEPS}")

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound
            print(f"CONST: {CONST}")
            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_watermark: self.wm})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, loss1, l2s, loss3, scores, nimg = self.sess.run([self.train, self.loss, self.loss1, self.l2dist,
                                                                       self.loss3, self.output, self.newimg])
                # attack done
                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(f"    Iteration {iteration} of {self.MAX_ITERATIONS} loss={l} "
                                  f"loss1={loss1} l2={np.mean(l2s)} loss3={loss3}") # scores={np.argmax(scores, axis=-1)}

                # check if we should abort search if we're getting nowhere.
                # if self.ABORT_EARLY and iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                #     if l > prev * .9999:
                #         _logger.debug("    Failed to make progress; stop early")
                #         break
                #     prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    lab = np.argmax(batchlab[e], axis=-1)
                    sc = np.argmax(sc, axis=-1)
                    o_bestattack[e] = ii
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = sc

                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = sc
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e], axis=-1)):  # and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

            _logger.debug("  Successfully generated adversarial examples on {} of {} instances.".format(sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
        # return nimg[0]
