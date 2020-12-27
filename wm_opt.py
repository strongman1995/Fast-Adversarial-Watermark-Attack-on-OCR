import time, io, sys
from tqdm import tqdm
import tensorflow as tf
import sklearn
from PIL import Image
import numpy as np
import pickle
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


img_data_path = '/home/chenlu/research/TextRecognitionDataGenerator/word_image_data'
with open(f'{img_data_path}/intersect_word.pkl', 'rb') as f:
    intersect_word = pickle.load(f)
intersect_word = set(intersect_word)

def filter_word(input_img, len_x, gt_txt, target_txt, wm0_mask):
    new_input_img, new_len_x, new_gt_txt, new_target_txt, new_wm0_mask = [], [], [], [], []
    for i, gt in enumerate(gt_txt):
        if gt in intersect_word:
            new_input_img.append(input_img[i])
            new_len_x.append(len_x[i])
            new_gt_txt.append(gt_txt[i])
            new_target_txt.append(target_txt[i])
            new_wm0_mask.append(wm0_mask[i])
    return np.asarray(new_input_img), new_len_x, new_gt_txt, new_target_txt, np.asarray(new_wm0_mask)

font_name, case, nb_iter = sys.argv[1], sys.argv[2], sys.argv[3]
# our_const = float(sys.argv[4])
with open(f'{img_data_path}/{font_name}-{case}.pkl', 'rb') as f:
    _, _, len_x, gt_txt, target_txt = pickle.load(f)

title = f"{font_name}-{case}-l2-eps0.2-ieps5.0-iter{nb_iter}"
with open(f'wm_result/{title}.pkl', 'rb') as f:
# with open(f'sample_images_calamari/grad-wm-{title}.pkl', 'rb') as f:
    (_, wm0_mask, _, wm0_img, _, _, _, _, _, _, _) = pickle.load(f)
input_img, len_x, gt_txt, target_txt, wm_mask_img = filter_word(wm0_img, len_x, gt_txt, target_txt, wm0_mask)
# input_img, len_x, gt_txt, target_txt, wm_mask_img = wm0_img[:100], len_x[:100], gt_txt[:100], target_txt[:100], wm0_mask[:100]

print(f"{font_name}-{case} {len(input_img)} examples.")

LEARNING_RATE = learning_rate = 0.01 # float
ABORT_EARLY = abort_early = True # bool
batch_size = 100 # int
clip_min, clip_max = 0.0, 1.0
np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')
shape = tuple([batch_size] + list(input_img.shape[1:]))  # (batch_size, height, width, channel)

# build graph
with graph.as_default():
    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))  # (batch_size, height, width, channel)
    # these are variables to be more efficient in sending data to tf
    timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
    input_seq_len = tf.Variable(np.zeros(batch_size), dtype=tf.int32, name='seq_len')
    wm_mask = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='wm_mask')
    const = tf.Variable(np.zeros(batch_size), dtype=tf_dtype, name='const')
    # and here's what we use to assign them
    assign_timg, assign_input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
    assign_wm_mask = tf.placeholder(tf_dtype, shape, name='assign_const')
    assign_const = tf.placeholder(tf_dtype, [batch_size], name='assign_const')

    # the resulting instance, tanh'd to keep bounded from clip_min to clip_max
    newimg = (tf.tanh(wm_mask * modifier + timg) + 1) / 2
    newimg = newimg * (clip_max - clip_min) + clip_min
    # prediction BEFORE-SOFTMAX of the model
    output_seq_len, time_major_logits, time_major_softmax, logits, softmax, decoded, sparse_decoded, scale_factor, \
    log_prob = network.create_network(newimg, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE)


    # sum up the losses
    loss1 = tf.nn.ctc_loss(labels=targets,
                           inputs=time_major_logits,
                           sequence_length=output_seq_len,
                           time_major=True,
                           ctc_merge_repeated=True,
                           ignore_longer_outputs_than_inputs=True)
    loss1 = tf.reduce_sum(const * loss1)  # mu * ctc_loss

    # distance to the input data
    other = (tf.tanh(timg) + 1) / 2 * (clip_max - clip_min) + clip_min
    l2dist = tf.reduce_sum(tf.square(newimg - other), axis=list(range(1, len(shape))))  # (batch_size, )
    loss2 = tf.reduce_sum(l2dist)  # L2-norm distance

    # nps: non-printability score
    closest_val = tf.divide(tf.round(newimg * 255), 255)  # 乘 255 取整后再除回去
    loss3 = tf.reduce_sum(10 * tf.abs(newimg - closest_val))

    # tv: total variation
    n_x, n_y = input_img.shape[1], input_img.shape[2]
    tmp1 = modifier[:, :n_x - 1, :n_y - 1]
    tmp2 = modifier[:, 1:n_x, :n_y - 1]
    tmp3 = modifier[:, :n_x - 1, 1:n_y]
    tv = tf.sqrt(tf.square(tmp1 - tmp2) + tf.square(tmp1 - tmp3) + 1e-5)
    loss4 = tf.reduce_sum(1e-2 * tv)

    loss = loss1 + loss2 + loss4

    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss, var_list=[modifier])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]  # 优化在 optimizer 和 train 里的变量

    # these are the variables to initialize when we run
    setup = []
    setup.append(timg.assign(assign_timg))
    setup.append(input_seq_len.assign(assign_input_seq_len))
    setup.append(wm_mask.assign(assign_wm_mask))
    setup.append(const.assign(assign_const))

    init = tf.variables_initializer(var_list=[modifier] + new_vars)


# preprocess img
imgs = input_img[:len(input_img) // batch_size * batch_size]
imgs = (imgs - clip_min) / (clip_max - clip_min) # re-scale instances to be within range [0, 1]
imgs = np.clip(imgs, clip_min, clip_max)
imgs = (imgs * 2) - 1 # now convert to [-1, 1]
imgs = np.arctanh(imgs * .999999) # convert to tanh-space

MAX_ITERATIONS = 1000
BINARY_SEARCH_STEPS = 1
initial_const = 10  # float
# initial_const = our_const
with graph.as_default():
    adv_img_list = []
    adv_l2_list = []
    adv_txt_list = []
    adv_iter_list = []
    adv_asr_list = []
    start = time.time()
    for i in tqdm(range(0, len(imgs), batch_size)):  # run attack in batch data
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)  # (batch_size, )
        CONST = np.ones(batch_size) * initial_const  # (batch_size, )
        upper_bound = np.ones(batch_size) * 1e10  # (batch_size, )

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size  # (batch_size, )
        o_bestscore = [-1] * batch_size  # (batch_size, )
        o_bestattack = np.zeros(shape)
        o_bestiter = [-1] * batch_size

        for outer_step in range(BINARY_SEARCH_STEPS):  # 二分调整 const
            # completely reset adam's internal state.
            sess.run(init)
            batch = imgs[i:i + batch_size]
            batch_len_x = len_x[i:i + batch_size]
            batch_wm_mask = wm_mask_img[i:i + batch_size]
            batch_target_txt = target_txt[i:i + batch_size]
            batch_tmp_y = [
                np.asarray([c - 1 for c in encode(t)])
                for t in batch_target_txt
            ]
            batch_y = sparse_tuple_from(batch_tmp_y)

            bestl2 = [1e10] * batch_size  # (batch_size, )
            bestscore = [-1] * batch_size  # (batch_size, )
            bestiter = [-1] * batch_size
            bestasr = [-1] * MAX_ITERATIONS
            # print(f"Binary search step {outer_step} of {BINARY_SEARCH_STEPS}")

            # set the variables so that we don't have to send them over again
            sess.run(
                setup, {
                    assign_timg: batch,
                    assign_input_seq_len: batch_len_x,
                    assign_wm_mask: batch_wm_mask,
                    assign_const: CONST,
                })

            for iteration in range(MAX_ITERATIONS):  # 开始迭代攻击
                # perform the attack
                _, l, scores, l2s, batch_adv_txt, nimg = sess.run(
                    [train, loss, loss1, l2dist, decoded, newimg],
                    feed_dict={
                        targets: batch_y,
                        dropout_rate: 0
                    })
                batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_txt)
                batch_adv_txt = [''.join(decode(index)) for index in batch_adv_index]

                # adjust the best result found so far
                cnt = 0
                for e, (l2, ii) in enumerate(zip(l2s, nimg)):
                    target_t, adv_t = batch_target_txt[e], batch_adv_txt[e]
                    if target_t == adv_t:
                        cnt += 1
                    if l2 < bestl2[e] and target_t == adv_t:
                        bestl2[e] = l2
                        bestscore[e] = adv_t
                        bestiter[e] = iteration
                    if l2 < o_bestl2[e] and target_t == adv_t:
                        o_bestl2[e] = l2
                        o_bestscore[e] = adv_t  # batch_adv_txt
                        o_bestattack[e] = ii
                        o_bestiter[e] = iteration
                bestasr[iteration] = cnt

                # check if we should abort search if we're getting nowhere.
                if ABORT_EARLY:
                    n_success = 0
                    for target_t, adv_t in zip(batch_target_txt, bestscore):
                        if target_t == adv_t:
                            n_success += 1
                    if n_success == len(batch_target_txt):
                        print(f"{font_name}-{case} [{iteration}] break, all attacks succeed!")
                        break

            # adjust the constant as needed
            for e in range(batch_size):  # 二分调整 const
                if bestscore[e] == batch_target_txt[e]:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    # or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

            print(
                f"Success {sum(upper_bound < 1e9)} of {batch_size} instances."
            )
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            print(f"Mean successful distortion: {mean:.4g}") # mean distortion in the success examples

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        adv_l2_list.append(o_bestl2)
        adv_txt_list += o_bestscore
        adv_img_list.append(o_bestattack)  # 把 batch adv img 加到 list 中
        adv_iter_list += o_bestiter
        adv_asr_list.append(bestasr)
        print('-' * 30, i, '-' * 30)

adv_img = np.asarray(adv_img_list).reshape(imgs.shape)
adv_l2 = np.asarray(adv_l2_list).reshape(-1)

title = f"{font_name}-{case}-smooth"
# title = f"{font_name}-{case}-{our_const}"
with open(f'wm_opt_result/{title}.pkl', 'wb') as f:
# with open(f'sample_images_calamari/opt-wm-{title}.pkl', 'wb') as f:
    pickle.dump((adv_img, adv_txt_list, adv_l2, adv_iter_list, adv_asr_list, time.time() - start), f)