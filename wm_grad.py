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


# build graph
with graph.as_default():
    # _ 是data_iterator如果是dataset input的话
    inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders(
    )
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

    # watermark mask
    mask = tf.placeholder(tf.float32,
                          shape=inputs.get_shape().as_list(),
                          name="mask")
    grad = tf.multiply(acc_m, mask, name="mask_op")
    # eps_iter = 0.01
    # ord=np.inf
    optimal_perturbation = tf.sign(grad)
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)
    # ord=1
    abs_grad = tf.abs(grad)
    max_abs_grad = tf.reduce_max(abs_grad, axis=red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, axis=red_ind, keepdims=True)
    optimal_perturbation = tf.sign(grad) * tied_for_max / num_ties
    scaled_perturbation_1 = utils_tf.mul(0.01, optimal_perturbation)
    # ord=2
    square = tf.maximum(
        1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
    scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)

# load image data
font_name, case, pert_type, eps, eps_iter, nb_iter = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
img_data_path = '/home/chenlu/research/TextRecognitionDataGenerator/word_image_data'
# img_data_path = '/home/chenlu/research/TextRecognitionDataGenerator/intersect_image_data'
# img_data_path = '/home/chenlu/research/TextRecognitionDataGenerator/paragraph_image_data/'
with open(f'{img_data_path}/{font_name}-{case}.pkl', 'rb') as f:
    img_list, input_img, len_x, gt_txt, target_txt = pickle.load(f)
input_img = np.asarray(input_img)
# input_img, len_x, gt_txt, target_txt = input_img[:100], len_x[:100], gt_txt[:100], target_txt[:100]

title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
# title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps5.0-iter{nb_iter}"
with open(f'attack_result/{title}.pkl', 'rb') as f:
# with open(f'sample_images_calamari/grad-basic-{title}.pkl', 'rb') as f:
    adv_img, record_adv_text, record_iter, (duration, total_iter) = pickle.load(f)
# adv_img, record_adv_text, record_iter = adv_img[:100], record_adv_text[:100], record_iter[:100]


from skimage import morphology
import cv2


def find_wm_pos(adv_img, input_img, ret_frame_img=False):
    pert = np.abs(cvt2raw(adv_img) - cvt2raw(input_img))
    pert = (pert > 1e-2) * 255.0
    wm_pos_list = []
    frame_img_list = []
    for src in pert:
        kernel = np.ones((3, 3), np.uint8)  # 设置卷积核3*3
        dilate = cv2.dilate(src, kernel, iterations=2)  # 图像的膨胀
        erode = cv2.erode(dilate, kernel, iterations=2)  # 图像的腐蚀
        remove = morphology.remove_small_objects(erode.astype('bool'), min_size=0)
        contours, _ = cv2.findContours((remove * 255).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        wm_pos, frame_img = [], []
        for cont in contours:
            left_point = cont.min(axis=1).min(axis=0)
            right_point = cont.max(axis=1).max(axis=0)
            wm_pos.append(np.hstack((left_point, right_point)))
            if ret_frame_img:
                img = cv2.rectangle(
                    (remove * 255).astype('uint8'), (left_point[0], left_point[1]),
                    (right_point[0], right_point[1]), (255, 255, 255), 2)
                frame_img.append(img)
        wm_pos_list.append(wm_pos)
        frame_img_list.append(frame_img)

    if ret_frame_img:
        return (wm_pos_list, frame_img_list)
    else:
        return wm_pos_list

pos, frames = find_wm_pos(adv_img, input_img, True)

# 按面积大小把pos从大到小排个序
new_pos = []
for _pos in pos:
    if len(_pos) > 1:
        new_pos.append(sorted(_pos, key=lambda x: (x[3]-x[1])*(x[2]-x[0]), reverse=True))
    else:
        new_pos.append(_pos)
pos = new_pos

from trdg.generators import GeneratorFromStrings

def gen_wm(RGB):
    generator = GeneratorFromStrings(
        strings=['eccv'],
        count=1,  # 五种字体
        fonts=['Impact.ttf'],  # default: []
        language='en',
        size=100,  # 32
        skewing_angle=10,
        random_skew=False,
        blur=0,
        random_blur=False,
        # gaussian noise (0), plain white (1), quasicrystal (2) or picture (3)
        background_type=1,
        distorsion_type=0,  # None(0), Sine wave(1),Cosine wave(2),Random(3)
        distorsion_orientation=0,
        is_handwritten=False,
        width=-1,
        alignment=1,
        text_color=RGB2Hex(RGB),
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(0, 0, 0, 0),
        fit=True,
    )
    img_list = [img for img, _ in generator]
    return img_list[0]


# RGB格式颜色转换为16进制颜色格式
def RGB2Hex(RGB): # RGB is a 3-tuple
    color = '#'
    for num in RGB:
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

# 得到水印 mask
grayscale = 0 # 灰度值在 76-226 之间有对应的彩色水印值
color = (grayscale, grayscale, grayscale)
wm_img = gen_wm(color)
wm_arr = np.array(wm_img.convert('L'))
kernel = np.ones((5, 5), np.uint8)
wm_arr = cv2.dilate(wm_arr, kernel, 2)
wm_arr = cv2.erode(wm_arr, kernel, 2)
bg_mask = ~(wm_arr != 255)

# 灰色水印
# grayscale = int(sys.argv[7])
grayscale = 174 # 灰度值在 76-226 有对应的彩色水印值，为了增加扰动后还在范围内，128-174
color = (grayscale, grayscale, grayscale)
wm_img = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
wm_img[bg_mask] = 255
wm_img = Image.fromarray(wm_img)

# large_l = []
# green_val = np.array(list(range(256)))
# for _gi in range(256):
#     _r, _g, _b = 255, _gi, 0
#     large_l.append(_r * 19595 + _g * 38470 + _b * 7471 + 0x8000)
# normal_l = np.array(large_l, dtype='uint32') >> 16
# l0_1 = normal_l / 255
# range_min, range_max = l0_1.min(), l0_1.max()
# gray_green_map = dict(list(zip(normal_l, green_val)))
# gray_green_map_array = np.ones((255, ))
# gray_green_map_array[:76] = gray_green_map[76]
# gray_green_map_array[227:] = gray_green_map[226]
# for gray, green in gray_green_map.items():
#     gray_green_map_array[gray] = green

# 彩色水印
# green_v = gray_green_map[grayscale]
# color = (255, green_v, 0)
# wm_img = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
# wm_img[bg_mask] = 255
# wm_img = Image.fromarray(wm_img)

def get_text_mask(img: np.array):
    if img.max() <= 1:
        return img < 1 / 1.25
    else:
        return img < 255 / 1.25


wm0_img_list = []
wm_mask_list = []
text_mask_list = []
for i in range(len(input_img)):
    text_img = show(input_img[i])
    text_mask = get_text_mask(np.array(text_img))  # 得到 text 的 mask (bool)
    rgb_img = Image.new(mode="RGB", size=text_img.size, color=(255, 255, 255))
    p = -int(wm_img.size[0] * np.tan(10 * np.pi / 180))
    right_shift = 10
    xp = pos[i][0][0]+right_shift if len(pos[i]) != 0 else right_shift
    # xp = 0
    rgb_img.paste(wm_img, box=(xp, p))  # 先贴 wm
    wm_mask = (np.array(rgb_img.convert('L')) != 255)  # 得到 wm 的 mask(bool)
    rgb_img.paste(text_img, mask=cvt2Image(text_mask))  # 再贴 text

    wm0_img_list.append(rgb_img)
    wm_mask_list.append(transpose(wm_mask))
    text_mask_list.append(transpose(text_mask))
wm_mask = np.asarray(wm_mask_list)
text_mask = np.asarray(text_mask_list)


batch_size = 100
clip_min, clip_max = 0.0, 1.0

# 大数据集查看
record_text = []
wm0_img = pred_img = np.asarray([cvt2raw(np.array(img.convert('L'))) / 255 for img in wm0_img_list])
batch_iter = len(input_img) // batch_size
batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1
for batch_i in range(batch_iter):
    batch_img = pred_img[batch_size * batch_i:batch_size * (batch_i + 1)]
    batch_len_x = len_x[batch_size * batch_i:batch_size * (batch_i + 1)]
    batch_text = sess.run(decoded,
                              feed_dict={
                                  inputs: batch_img,
                                  input_seq_len: batch_len_x,
                                  dropout_rate: 0,
                              })
    batch_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_text)
    record_text += [''.join(decode(index)) for index in batch_index]

cnt = 0
for pred_txt, raw_txt in zip(record_text, gt_txt):
    if pred_txt == raw_txt:
        cnt += 1

accuracy = cnt / len(gt_txt)

# run attack

target_index_list = [np.asarray([c for c in encode(t)]) for t in target_txt]
wm_img = wm0_img
with graph.as_default():
    adv_img = wm_img.copy()
    m0 = np.zeros(input_img.shape)
    record_iter = np.zeros(input_img.shape[0])  # 0代表没成功
    record_mse = []
    record_mse_plus = []
    start = time.time()
    for i in tqdm(range(nb_iter)):
        batch_iter = len(input_img) // batch_size
        batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1
        for batch_i in range(batch_iter):
            batch_input_img = wm_img[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_adv_img = adv_img[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_len_x = len_x[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_m0 = m0[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_target_txt = target_txt[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_tmp_y = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_txt]
            batch_y = sparse_tuple_from(batch_tmp_y)
            batch_mask = wm_mask[batch_size * batch_i:batch_size * (batch_i + 1)]
            batch_record_iter = record_iter[batch_size * batch_i:batch_size * (batch_i + 1)]

            scaled_perturbation = scaled_perturbation_2 if pert_type == '2' else scaled_perturbation_inf
            batch_pert = sess.run(scaled_perturbation,
                                  feed_dict={
                                      inputs: batch_adv_img,
                                      input_seq_len: batch_len_x,
                                      m: batch_m0,
                                      targets: batch_y,
                                      mask: batch_mask,
                                      dropout_rate: 0,
                                  })
            batch_pert[batch_record_iter != 0] = 0
            batch_adv_img = batch_adv_img + eps_iter * batch_pert  * (batch_pert > 0) # negative
            batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
            batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)
            adv_img[batch_size * batch_i:batch_size * (batch_i + 1)] = batch_adv_img
        record_mse.append(np.mean(((adv_img - wm_img) * 255) ** 2))
        record_mse_plus.append(np.mean((((adv_img - wm_img) * ((adv_img - wm_img) > 0)) * 255) ** 2))

        record_adv_text = []
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
            for j in (range(len(batch_target_index))):
                # attack img j successfully at iter i
                adv_index, target_index = batch_adv_index[j], batch_target_index[j]
                idx_j = batch_size * batch_i + j
                if np.sum(adv_index != target_index) == 0 and record_iter[idx_j] == 0:
                    record_iter[idx_j] = i
        if np.sum(record_iter == 0) == 0:  # all examples are successful
            break
    duration = time.time() - start
    print(f"{i} break. Time cost {duration:.4f} s")

def cvt2rgb(gray_img, text_mask):
    gray_img = invert(gray_img)
    op_mask = (~(gray_img == 1)) & (~text_mask) # not_bg & not_text
    rgb_img = np.ones(list(gray_img.shape)+[3])
    rgb_img[:, :, :, 0] = gray_img
    rgb_img[:, :, :, 1] = gray_img
    rgb_img[:, :, :, 2] = gray_img
    rgb_img[op_mask, 0] = 1
    rgb_img[op_mask, 1] = (gray_img[op_mask] - 0.299) / 0.587
    rgb_img[op_mask, 2] = 0
    return invert(rgb_img)

rgb_img = cvt2rgb(adv_img, text_mask)


# with open(f'wm_result/{font_name}-{case}-mse.pkl', 'wb') as f:
#     pickle.dump((record_mse, record_mse_plus), f)

# title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
# title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}-gray{int(sys.argv[7])}"
title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}-positive"
with open(f'wm_result/{title}.pkl', 'wb') as f:
# # with open(f'sample_images_calamari/grad-wm-{title}.pkl', 'wb') as f:
    pickle.dump((pos, wm_mask, text_mask, wm0_img, record_text, accuracy, adv_img, record_adv_text, record_iter, (duration, i), rgb_img), f)
