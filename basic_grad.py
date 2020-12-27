# -*- coding: utf-8 -*-
# @Time    : 13/1/20 15:31
# @Author  : Lu Chen
import tensorflow as tf
import sklearn
from PIL import Image
import numpy as np
import pickle, glob, time, sys, os
from tqdm import tqdm
from cleverhans import utils_tf
from util import get_argparse, cvt2Image, sparse_tuple_from
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel
from calamari_ocr.ocr import Predictor

# parse the parameters from shell
parser = get_argparse()
args = parser.parse_args()

predictor = Predictor(checkpoint=os.path.join("ocr_model", args.model_path), batch_size=1, processes=10)
network = predictor.network
sess, graph = network.session, network.graph
encode, decode = network.codec.encode, network.codec.decode

# build graph
with graph.as_default():
    # _ 是data_iterator如果是dataset input的话
    inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
    output_seq_len, time_major_logits, time_major_softmax, logits, softmax, decoded, sparse_decoded, scale_factor, log_prob = \
        network.create_network(inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE)
    loss = tf.nn.ctc_loss(labels=targets,
                          inputs=time_major_logits,
                          sequence_length=output_seq_len,
                          time_major=True,
                          ctc_merge_repeated=True,
                          ignore_longer_outputs_than_inputs=True)
    loss = -tf.reduce_mean(loss, name='loss')
    grad, = tf.gradients(loss, inputs)

    # Normalize current gradient and add it to the accumulated gradient
    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = tf.cast(1e-12, grad.dtype)
    divisor = tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True)
    norm_grad = grad / tf.maximum(avoid_zero_div, divisor)

    m = tf.placeholder(tf.float32,
                       shape=inputs.get_shape().as_list(),
                       name="momentum")
    acc_m = m + norm_grad

    grad = acc_m
    # ord=np.inf
    optimal_perturbation = tf.sign(grad)
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)
    # ord=1
    # abs_grad = tf.abs(grad)
    # max_abs_grad = tf.reduce_max(abs_grad, axis=red_ind, keepdims=True)
    # tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    # num_ties = tf.reduce_sum(tied_for_max, axis=red_ind, keepdims=True)
    # optimal_perturbation = tf.sign(grad) * tied_for_max / num_ties
    # scaled_perturbation_1 = utils_tf.mul(0.01, optimal_perturbation)
    # ord=2
    square = tf.maximum(1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
    scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)

# set parameters
font_name = args.font_name
case = args.case
pert_type = args.pert_type
eps = args.eps
eps_iter = args.eps_iter
nb_iter = args.nb_iter
batch_size = args.batch_size
clip_min, clip_max = args.clip_min, args.clip_max

# load img data
with open(f'img_data/{font_name}.pkl', 'rb') as f:
    input_img, len_x, gt_txt = pickle.load(f)
# load attack pair
with open(f'attack_pair/{font_name}-{case}.pkl', 'rb') as f:
    _, target_txt = pickle.load(f)

# small samples
n_img = 200
input_img, len_x, gt_txt, target_txt = input_img[:n_img], len_x[:n_img], gt_txt[:n_img], target_txt[:n_img]

# run attack
with graph.as_default():
    adv_img = input_img.copy()
    m0 = np.zeros(input_img.shape)
    record_iter = np.zeros(input_img.shape[0])  # 0 stands for unsuccess
    record_adv_text = []
    # perform attack in batch images
    batch_iter = len(input_img) // batch_size
    batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1
    start = time.time()
    for batch_i in tqdm(range(batch_iter)):
        batch_input_img = input_img[batch_size * batch_i:batch_size * (batch_i + 1)]
        batch_adv_img = adv_img[batch_size * batch_i:batch_size * (batch_i + 1)]
        batch_len_x = len_x[batch_size * batch_i:batch_size * (batch_i + 1)]
        batch_m0 = m0[batch_size * batch_i:batch_size * (batch_i + 1)]
        batch_target_text = target_txt[batch_size * batch_i:batch_size * (batch_i + 1)]
        batch_target_index = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_text]
        batch_y = sparse_tuple_from(batch_target_index)
        batch_record_iter = record_iter[batch_size * batch_i:batch_size * (batch_i + 1)]

        scaled_perturbation = scaled_perturbation_2 if pert_type == '2' else scaled_perturbation_inf

        batch_record_iter = np.zeros(batch_size)
        for i in (range(nb_iter)):
            batch_pert, batch_adv_text = sess.run(
                [scaled_perturbation, decoded],  # pert type
                feed_dict={
                    inputs: batch_adv_img,
                    input_seq_len: batch_len_x,
                    m: batch_m0,
                    targets: batch_y,
                    dropout_rate: 0,
                })
            batch_pert[batch_record_iter != 0] = 0
            batch_adv_img = batch_adv_img + eps_iter * batch_pert
            batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
            batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)
            adv_img[batch_size * batch_i:batch_size * (batch_i + 1)] = batch_adv_img

            batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_text)
            batch_adv_text = [''.join(decode(index)) for index in batch_adv_index]

            for j in range(batch_size):
                if batch_adv_text[j] == batch_target_text[j] and batch_record_iter[j] == 0:
                    batch_record_iter[j] = i
            # check whether all batch examples are successful
            if np.sum(batch_record_iter == 0) == 0:
                print(f"{i} break")
                break
        record_iter[batch_size * batch_i:batch_size * (batch_i + 1)] = batch_record_iter
        record_adv_text += batch_adv_text
    duration = time.time() - start

# save the attack result
title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
with open(f'attack_result/{title}.pkl', 'wb') as f:
    pickle.dump((adv_img, record_adv_text, record_iter, (duration, i)), f)
