import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
import tensorflow as tf
from calamari_ocr.ocr import Predictor
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel

checkpoint = '/home/chenlu/calamari/models/antiqua_modern/4.ckpt.json'
predictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)
network = predictor.network
sess = network.session
graph = network.graph

# build graph
with graph.as_default():
    inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
    output_seq_len, time_major_logits, time_major_softmax, logits, softmax, decoded, sparse_decoded, scale_factor, log_prob = \
        network.create_network(inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE)

import pickle
import glob
from tqdm import tqdm

fonts_files = glob.glob('/home/chenlu/research/TextRecognitionDataGenerator/fonts/**', recursive=True)
fonts = []
for file in fonts_files:
    if file.endswith('ttf'):
        fonts.append(file)

for font in fonts:
    save_name = font.split('/')[-1][:-4]
    print(save_name)
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/image_data/{save_name}.pkl', 'rb') as f:
        img_list, lbl_list = pickle.load(f)

    # remove images containing exceptional characters
    charset = set(network.codec.charset)
    new_img_list, new_lbl_list = [], []
    for i, text in enumerate(lbl_list):
        if set(text).issubset(charset):
            new_img_list.append(img_list[i])
            new_lbl_list.append(lbl_list[i])
    img_list, lbl_list = new_img_list, new_lbl_list

    # preprocess image data
    images = [np.array(img_list[i].convert('L'), dtype='uint8') for i in range(len(img_list))]
    images, params = zip(*predictor.data_preproc.apply(images))
    images, len_x = network.zero_padding(images)  # padding images to same fixed-length images
    images = images / 255  # normalized images
    input_img = images.reshape(images.shape[:3])

    with graph.as_default():
        text_list = []
        batch_size = 100
        for i in tqdm(range(len(input_img) // batch_size + 1)):
            if i == len(input_img) // batch_size:
                decoded0 = sess.run(decoded,
                                    feed_dict={
                                        inputs: input_img[batch_size * i:],
                                        input_seq_len: len_x[batch_size * i:],
                                        dropout_rate: 0,
                                    })
            else:
                decoded0 = sess.run(decoded,
                                    feed_dict={
                                        inputs: input_img[batch_size * i:batch_size * (i + 1)],
                                        input_seq_len:
                                            len_x[batch_size * i:batch_size * (i + 1)],
                                        dropout_rate: 0,
                                    })
            labels = TensorflowModel._TensorflowModel__sparse_to_lists(decoded0)
            text = [''.join(network.codec.decode(label)) for label in labels]
            text_list += text
        # decoded0 = sess.run(decoded,
        #                     feed_dict={
        #                         inputs: input_img,
        #                         input_seq_len: len_x,
        #                         dropout_rate: 0,
        #                     })
        # labels = TensorflowModel._TensorflowModel__sparse_to_lists(decoded0)
        # text = [''.join(network.codec.decode(label)) for label in labels]
        # text_list += text

    # remove data that pred_text is not equal to raw_text
    new_img_list, new_lbl_list, new_input_img, new_len_x = [], [], [], []
    bad_img_list, bad_lbl_list, bad_input_img, bad_len_x = [], [], [], []
    for i, (pred_text, raw_text) in enumerate(zip(text_list, lbl_list)):
        if pred_text == raw_text:
            new_img_list.append(img_list[i])
            new_lbl_list.append(lbl_list[i])
            new_input_img.append(input_img[i])
            new_len_x.append(len_x[i])
        else:
            bad_img_list.append(img_list[i])
            bad_lbl_list.append(lbl_list[i])
            bad_input_img.append(input_img[i])
            bad_len_x.append(len_x[i])

    # save to pickle file
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}.pkl', 'wb') as f:
        pickle.dump((new_img_list, new_lbl_list), f)

    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}-bad.pkl', 'wb') as f:
        pickle.dump((bad_img_list, bad_lbl_list), f)

    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}-input.pkl', 'wb') as f:
        pickle.dump((np.array(new_input_img), new_lbl_list, new_len_x), f)

    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}-input-bad.pkl', 'wb') as f:
        pickle.dump((np.array(bad_input_img), bad_lbl_list, bad_len_x), f)


##################  find intersection of sentences in different fonts(only norm + bd)

sent_list = []
for font in fonts:
    save_name = font.split('/')[-1][:-4]
    print(save_name)
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}.pkl', 'rb') as f:
        _, lbl_list = pickle.load(f)
        sent_list.append(lbl_list)

norm = [3, 6, 10, 12]
bd = [1, 4, 7, 9, 13]
li = [2, 5, 8, 11, 14]

intersection = set(sent_list[0])
idx_list = norm + bd
for i in idx_list:
    intersection = intersection & set(sent_list[i])

with open('/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/intersection_set.pkl', 'wb') as f:
    pickle.dump(intersection, f)

################# save intersection image data of difference fonts (include norm+bd+li?????)

for font in fonts:
    save_name = font.split('/')[-1][:-4]
    print(save_name)
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}.pkl', 'rb') as f:
        img_list, lbl_list = pickle.load(f)
    new_img_list, new_lbl_list = [], []
    for i, sent in enumerate(lbl_list):
        if sent in intersection:
            new_img_list.append(img_list[i])
            new_lbl_list.append(lbl_list[i])

    # save to pickle file
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/intersect_image_data/{save_name}.pkl', 'wb') as f:
        pickle.dump((new_img_list, new_lbl_list), f)

    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/new_image_data/{save_name}-input.pkl', 'rb') as f:
        img_list, lbl_list, len_x = pickle.load(f)
    new_img_list, new_lbl_list, new_len_x = [], [], []
    for i, sent in enumerate(lbl_list):
        if sent in intersection:
            new_img_list.append(img_list[i])
            new_lbl_list.append(lbl_list[i])
            new_len_x.append(len_x[i])

    # save to pickle file
    with open(f'/home/chenlu/research/TextRecognitionDataGenerator/intersect_image_data/{save_name}-input.pkl', 'wb') as f:
        pickle.dump((np.asarray(new_img_list), new_lbl_list, new_len_x), f)
