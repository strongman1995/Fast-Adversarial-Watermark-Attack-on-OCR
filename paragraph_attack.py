from tqdm import tqdm
import tensorflow as tf
import sklearn
from PIL import Image
import numpy as np
import pickle, glob, time, sys, io
from cleverhans import utils_tf
from util import cvt2Image, sparse_tuple_from
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel
from calamari_ocr.ocr import Predictor
checkpoint = '/home/chenlu/calamari/models/antiqua_modern/4.ckpt.json'
predictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)

network = predictor.network
sess, graph = network.session, network.graph
codec = network.codec
charset = codec.charset
encode, decode = codec.encode, codec.decode
code2char, char2code = codec.code2char, codec.char2code

def invert(data): # 反色
    if data.max() < 1.5:
        return 1 - data
    else:
        return 255 - data

def transpose(data): # 旋转90度
    if len(data.shape) != 2:
        return np.swapaxes(data, 1, 2)
    else:
        return data.T

def cvt2raw(data):
    return transpose(invert(data))

def show(img):
    return cvt2Image(cvt2raw(img))

def preprocess(img_list):
    # preprocess the image before feeding it into the model
    images = [np.array(img_list[i].convert('L'), dtype='uint8') for i in range(len(img_list))]
    images, params = zip(*predictor.data_preproc.apply(images))
    images, len_x = network.zero_padding(images)  # padding images to same fixed-length images
    images = images / 255 # normalized images
    input_img = images.reshape(images.shape[:3])
    return input_img, len_x

eps, eps_iter, nb_iter = 0.2, 5.0, 1000
batch_size = 100
clip_min, clip_max = 0.0, 1.0
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
    # ord=2
    square = tf.maximum(1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
    scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)

def attack(input_img, len_x, target_txt, pert_type='2'):
    target_index_list = [np.asarray([c for c in encode(t)]) for t in target_txt]
    with graph.as_default():
        adv_img = input_img.copy()
        m0 = np.zeros(input_img.shape)
        record_iter = np.zeros(input_img.shape[0])  # 0代表没成功

        start = time.time()
        for i in tqdm(range(nb_iter)):
            # perform attack
            batch_iter = len(input_img) // batch_size
            batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1
            for batch_i in range(batch_iter):
                batch_input_img = input_img[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_adv_img = adv_img[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_len_x = len_x[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_m0 = m0[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_target_txt = target_txt[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_tmp_y = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_txt]
                batch_y = sparse_tuple_from(batch_tmp_y)
                batch_record_iter = record_iter[batch_size * batch_i:batch_size * (batch_i + 1)]

                scaled_perturbation = scaled_perturbation_2 if pert_type == '2' else scaled_perturbation_inf
                batch_pert = sess.run(scaled_perturbation, # pert type
                                      feed_dict={
                                          inputs: batch_adv_img,
                                          input_seq_len: batch_len_x,
                                          m: batch_m0,
                                          targets: batch_y,
                                          dropout_rate: 0,
                                      })
                batch_pert[batch_pert > 0] = 0 ###########################3
                batch_pert[batch_record_iter != 0] = 0
                batch_adv_img = batch_adv_img + eps_iter * batch_pert
                batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
                batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)
                adv_img[batch_size * batch_i:batch_size * (batch_i + 1)] = batch_adv_img

            record_adv_text = []
            # check whether attack success
            for batch_i in range(batch_iter):
                batch_adv_img = adv_img[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_len_x = len_x[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_target_index = target_index_list[batch_size * batch_i:batch_size * (batch_i + 1)]
                batch_adv_text = sess.run(decoded,
                                          feed_dict={
                                              inputs: batch_adv_img,
                                              input_seq_len: batch_len_x,
                                              dropout_rate: 0,
                                          })
                batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_text)
                record_adv_text += [''.join(decode(index)) for index in batch_adv_index]
                for j in range(len(batch_target_index)):
                    # attack img idx_j successfully at iter i
                    idx_j = batch_size * batch_i + j
                    adv_index, target_index = batch_adv_index[j], batch_target_index[j]
                    if np.sum(adv_index != target_index) == 0 and record_iter[idx_j] == 0:
                        record_iter[idx_j] = i
            # check whether all examples are successful
            if np.sum(record_iter == 0) == 0:
                break

        duration = time.time() - start
        print(f"{i} break. Time cost {duration:.4f} s")
    return adv_img, record_adv_text, record_iter, (duration, i)

# paragraph
data_path = '/home/chenlu/research/TextRecognitionDataGenerator/paragraph_image_data/'
font = sys.argv[1]
with open(f'{data_path}/{font}.pkl', 'rb') as f:
    img_list, line_img_list, gt_txt = pickle.load(f)

width, height = line_img_list[0][0].size

line_img = []
for line_list in line_img_list:
    line_img += line_list

line_text = []
for g_txt in gt_txt:
    line_text += g_txt

len_x = [img.size[0] for img in line_img]
nb_line = len(line_img)
width = max(len_x)
input_img = (np.ones((nb_line, height, width)) * 255).astype('uint8')
for row, ih in enumerate(range(nb_line)):
    input_img[row, :, :line_img[row].size[0]] = np.array(line_img[row])
input_img = np.swapaxes(1 - (input_img / 255), 1, 2)
input_img, len_x = preprocess(line_img)

line_img_num = [len(line) for line in gt_txt]

record = [0]
for nb in line_img_num:
    record.append(record[-1] + nb)

cand_input_img = []
cand_len_x = []
cand_gt_txt = []
for i in range(len(line_img_num)):
    print(i)
    s, e = record[i], record[i+1]
    adv_img, record_adv_text, record_iter, (duration, i) = attack(input_img[s:e], len_x[s:e], line_text[s:e])
    if np.sum(record_iter == 0) == 0:
        cand_input_img.append(adv_img)
        cand_len_x.append(len_x[s:e])
        cand_gt_txt.append(line_text[s:e])
input_img, len_x, gt_txt = cand_input_img, cand_len_x, cand_gt_txt

# load English dictionary en_list
from trdg.utils import load_dict
en_list = load_dict('en_alpha') # 只包括字母的单词

# 将English dictionary中的word按照长度分类 en_dict
from collections import defaultdict
en_dict = defaultdict(list)
for w in en_list:
    en_dict[len(w)].append(w.lower())

import random
def find_new_word(w):
    new_w = random.choice(en_dict[len(w)])
    if w.istitle():
        new_w = new_w[0].upper() + new_w[1:]
    return new_w

import re
target_txt = []
for gt_p in gt_txt:
    target_i = []
    for gt in gt_p:
        sent = gt.split(' ')
        target_t = gt
        for k, w in enumerate(sent):
            if 4 <= len(w) <= 6 and re.match(r'^[a-z]*$', w.lower()):
                target_t = ' '.join(sent[:k] + [find_new_word(w)] + sent[k + 1:])
                break
        target_i.append(target_t)
    target_txt.append(target_i)

def extend(input):
    tmp_list = []
    for i in input:
        tmp_list += list(i)
    return tmp_list

# input_img = np.asarray(extend(input_img))
input_img_list = []
for imgs in input_img:
    for i in imgs:
        input_img_list.append(i)
input_img = np.asarray(input_img_list)

len_x = extend(len_x)
gt_txt = extend(gt_txt)
target_txt = extend(target_txt)

with open(f'{data_path}/{font}-new.pkl', 'wb') as f:
    pickle.dump((None, input_img, len_x, gt_txt, target_txt), f)
