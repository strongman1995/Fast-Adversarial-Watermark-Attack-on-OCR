# coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys, os
import random
import argparse


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


def clip(x, clip_min, clip_max):
    fix_min = x < clip_min
    x[fix_min] = clip_min
    fix_max = x > clip_max
    x[fix_max] = clip_max
    return x


def eval_distortion(batch_x, batch_adv):
    eta = abs(batch_x - batch_adv)
    return np.mean(np.sum(eta ** 2, axis=(1, 2, 3)) / np.prod(eta.shape[1:]))


def eval_asr(y_true, y_target):
    nb_samples = y_target.shape[0]
    nb_equal = 0
    for i in range(nb_samples):
        if np.argmax(y_true[i]) == np.argmax(y_target[i]):
            nb_equal += 1
    return nb_equal / nb_samples


def plot(x, y1, y2, x_lab, y1_lab, y2_lab, title, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y1, '-', label=y1_lab)
    ax2 = ax.twinx()
    ax2.plot(x, y2, '-r', label=y2_lab)
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y1_lab)
    ax2.set_ylabel(y2_lab)
    ax2.legend(loc=4)
    ax2.set_title(title)
    plt.savefig(f'result/{save_name}.png')


def load_similar():
    with open('sim_character.txt', 'r', encoding='utf-8') as f:
        similar_dict = dict()
        for line in f:
            ls = line.split(' ')
            similar_dict[ls[0]] = ls[1].strip()
    return similar_dict


def save(X, name="result"):
    if len(X.shape) == 2:
        X = trans_back(X).astype('uint8')
    elif len(X.shape) == 3:
        X = trans_back(X.reshape([X.shape[0], X.shape[1]])).astype('uint8')
    elif len(X.shape) == 4:
        X = trans_back(X.reshape([X.shape[1], X.shape[2]])).astype('uint8')
    Image.fromarray(X).save('images/{}.png'.format(name))


# plt.imsave("images/result.jpg", trans_back(X_adv).reshape([32, 682]))
# Image.fromarray(trans_back(X_adv).reshape([32, 682]).astype('uint8')).save('images/result.jpg')

def trans_back(X):
    return (X + 0.5) * 255


def dense_to_sparse(dense_tensor, sparse_val=0, out_type=tf.int64):
    """Inverse of tf.sparse_to_dense.
    Parameters: dense_tensor: The dense tensor. Duh.
                sparse_val: The value to "ignore": Occurrences of this value in the dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you will probably want to choose this as the default value.
    Returns: SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val), name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds, name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape", out_type=out_type)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args: sequences: a list of lists of type dtype where each element is a sequence
    Returns: A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    # 自动寻找序列的最大长度，形状为：batch_size * max_len
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
    # return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何
def getPixel(image, x, y, G, N):
    L = image.getpixel((x, y))
    if L > G:
        L = True
    else:
        L = False

    nearDots = 0
    if L == (image.getpixel((x - 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y + 1)) > G):
        nearDots += 1

    if nearDots < N:
        return image.getpixel((x, y - 1))
    else:
        return None


# 降噪
# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点
# G: Integer 图像二值化阀值
# N: Integer 降噪率 0 <N <8
# Z: Integer 降噪次数
# 输出
#  0：降噪成功
#  1：降噪失败
def clearNoise(image, G, N, Z):
    draw = ImageDraw.Draw(image)

    for i in range(0, Z):
        for x in range(1, image.size[0] - 1):
            for y in range(1, image.size[1] - 1):
                color = getPixel(image, x, y, G, N)
                if color != None:
                    draw.point((x, y), color)


def scale(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    img = np.array(img.resize([width, 32], Image.ANTIALIAS))

    if img.max() > 0.5:
        # scale to [-0.5, 0.5]
        img = img.astype(np.float32) / 255.0 - 0.5
        img = img.reshape([1, 32, width, 1])
    return img


def predict(img):
    if len(img.shape) == 2 or len(img.shape) == 3:
        return decode(model.predict(img.reshape([1, img.shape[0], img.shape[1], 1])))
    elif len(img.shape) == 4:
        return decode(model.predict(img))


def contrast_brightness_image(src1, a, g):
    if len(src1.shape) == 2:
        (h, w), ch = src1.shape, 1
    else:
        h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    return dst


# 定义添加椒盐噪声的函数
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random() > 0.5:
            SP_NoiseImg[randX, randY] = 0
        else:
            SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg


# 定义添加高斯噪声的函数
def addGaussianNoise(image, percetage):
    """
    add Gaussian Noise
    :param image: a 2D np.array
    :param percetage: the percent of added noise pixel. range [0, 1]
    :return:
    """
    G_Noiseimg = image
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for _ in range(G_NoiseNum):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg


def gen_text(label, fontsize=20, fontName="ocr_tensorflow_cnn/fonts/simhei.ttf", dir="images/", show=False):
    color = (0, 0, 0)
    img = Image.new(mode="RGBA", size=((int)(fontsize * len(label)), (int)(fontsize * 1.6)), color=(255, 255, 255))
    font = ImageFont.truetype(fontName, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text(xy=(0, 7), text=label, fill=color, font=font)
    save_path = os.path.join(dir, f"{label}-{fontsize}.png")
    img.save(save_path)
    if show:
        plt.imshow(np.array(Image.open(save_path)))
    return img


def gen_watermark(label,
                  fontsize=20,
                  fontName="ocr_tensorflow_cnn/fonts/DejaVuSansMono-Bold.ttf",
                  dir="images/",
                  show=False):
    color = (0, 0, 0)
    font_width = int(fontsize * (len(label) / 1.6))
    width = (int)(fontsize * (len(label) / 1.6) * 1.5)
    # height = (int)(fontsize*2)
    height = 32
    img = Image.new(mode="RGB", size=(width, height), color=(255, 255, 255))
    font = ImageFont.truetype(fontName, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text(((width - font_width) // 2, (height - fontsize) // 2),
              label,
              fill=color,
              font=font)
    save_path = os.path.join(dir, f"{label}-{fontsize}.jpg")
    img.save(save_path)
    if show:
        plt.imshow(np.array(Image.open(save_path)))
    return img


def gen_watermark_color(label,
                        grayscale=255,
                        fontsize=20,
                        fontName="ocr_tensorflow_cnn/fonts/DejaVuSansMono-Bold.ttf",
                        dir="images/",
                        show=False):
    color = (255, 255 - grayscale, 0)
    font_width = int(fontsize * (len(label) / 1.6))
    width = (int)(fontsize * (len(label) / 1.6) * 1.5)
    # height = (int)(fontsize*2)
    height = 32
    img = Image.new(mode="RGB", size=(width, height), color=(255, 255, 255))
    font = ImageFont.truetype(fontName, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text(((width - font_width) // 2, (height - fontsize) // 2),
              label,
              fill=color,
              font=font)
    save_path = os.path.join(dir, f"{label}-{fontsize}.jpg")
    img.save(save_path)
    if show:
        plt.imshow(np.array(Image.open(save_path)))
    return img


def gen_wm(bg_size,
           label,
           loc,
           font_size,
           gray=200,
           rotate=0,
           font_name="ocr_tensorflow_cnn/fonts/DejaVuSansMono-Bold.ttf",
           save_dir="images/",
           show=False):
    # font_width = int(fontsize * (len(label) / 1.6))
    wm = Image.new(mode="RGBA", size=bg_size, color=(255, 255, 255))
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(wm)
    draw.text(loc, label, fill=(gray, gray, gray), font=font)
    wm = wm.rotate(rotate, fillcolor=(255, 255, 255))
    return wm


def cvt2Image(array):
    if len(array.shape) == 3:
        array = array.reshape([array.shape[0], array.shape[1]])
    elif len(array.shape) == 4:
        array = array.reshape([array.shape[1], array.shape[2]])

    if array.max() <= 0.5:
        return Image.fromarray(((array + 0.5) * 255).astype('uint8'))
    elif array.max() <= 1:
        return Image.fromarray((array * 255).astype('uint8'))
    elif array.max() <= 255:
        return Image.fromarray(array.astype('uint8'))


def rotate_whole_img(img, angle):
    """
       rotate noise
       rotate angle is 0 - 20
    """
    if angle > 20:
        raise Exception("angle is 0 - 20")
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    im = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    return im


def attach_img(base_img, attach_img):
    h_start = (new_height - attach_img.shape[1]) // 2
    h_end = h_start + attach_img.shape[1]
    w_start = (new_width - attach_img.shape[2]) // 2
    w_end = w_start + attach_img.shape[2]
    base_img[:, h_start:h_end, w_start:w_end, :] = attach_img
    return base_img, (h_start, h_end, w_start, w_end)


def cvt4(img):
    return img.reshape([1, img.shape[0], img.shape[1], 1])

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help="Calamari-OCR model path.",
                        type=str)
    parser.add_argument("--font_name",
                        help="font name.",
                        type=str,
                        choices=['Courier',
                                 'Georgia',
                                 'Helvetica',
                                 'Times',
                                 'Arial'])
    parser.add_argument("--case",
                        help="case with different targets.",
                        type=str)
    parser.add_argument("--pert_type",
                        help="the bound type of perturbations",
                        type=str,
                        choices=['2', 'inf'])
    parser.add_argument("--eps",
                        help="perturbations is clipped by eps",
                        type=float)
    parser.add_argument("--eps_iter",
                        help="coefficient to adjust step size of each iteration",
                        type=float)
    parser.add_argument("--nb_iter",
                        help="number of maximum iteration",
                        type=int)
    parser.add_argument("--batch_size",
                        help="the number of samples per batch",
                        type=int)
    parser.add_argument("--clip_min",
                        help="the minimum value of images",
                        type=float)
    parser.add_argument("--clip_max",
                        help="the maximum value of images",
                        type=float)
    return parser
