# --*-- coding:utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import math
import itertools
import sys
reload(sys);sys.setdefaultencoding('utf-8')

def pad_box(box, ratio):
    x_, y_, w_, h_ = box
    x = max(0, int(x_ - w_ * ratio))
    y = max(0, int(y_ - h_ * ratio))
    w = w_ + int(ratio * 2 * w_)
    h = h_ + int(ratio * 2 * h_)
    return [x, y, w, h]

def crop_img(img, box):
    x,y,w,h = box
    return img[y:y+h, x:x+w]

def shift_box(main_box, sub_box):
    x, y = main_box[0], main_box[1]
    x0, y0, w0, h0 = sub_box
    new_box = [x0+x,y0+y,w0,h0]
    return new_box

def draw_box_string(img, box, string):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/home/gp/work/tools/useful/simhei.ttf", 20, encoding="utf-8")
    draw.text((x+w+20, y), string, (0, 255, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def draw_box(img, box, string):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    return img

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0
    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union

def is_night(img, thres=50):
    '''
    if the img is in night return true else return false.
    '''
    img_BGR = img.copy()
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    img_H, img_S, img_V = cv2.split(img_HSV)
    average_V = np.sum(np.reshape(img_V, (img_V.size,))) / img_V.size
    return average_V < thres

def img_merge(path_list, avi_dir):
    '''
    get demo.avi from path_list.
    '''
    video_path = '/mnt/mfs3/lizhilong/testuint/detectron_mask_rcnn/video.mp4'
    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.cv.CV_CAP_PROP_FPS)
    fps *= 0.2
    size = (int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G')  
    videoWriter = cv2.VideoWriter(avi_dir, fourcc, fps, size)
    for img_path in path_list:
        img = cv2.imread(img_path)
        videoWriter.write(img)
    videoWriter.release()

def get_path_list(img_dir):
    imgName_list = os.listdir(img_dir)
    path_list = [os.path.join(img_dir, imgName) for imgName in imgName_list]
    path_list.sort(key=lambda x:int(x.split("/")[-1].split(".")[0]), reverse=False)
    return path_list

def img_box_blur(img, box):
    '''
    blur img where in box.
    '''
    x,y,w,h = box
    head = img[y:y+h, x:x+w]
    head = cv2.GaussianBlur(head, (15,15), 125)
    img[y:y+h, x:x+w] = head
    return img

def vis_mask(img, box, col, alpha=0.7, show_border=False, border_thick=1):
    """
    Visualizes a single binary mask.
    col = np.array([0.2, 0.7, 0.4])*225
    """
    img = img.astype(np.float32)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    x,y,w,h = box 
    b = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype = np.int32)
    cv2.fillPoly(mask,[b], 1)
    idx = np.nonzero(mask)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col 
    if show_border:
        a = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, a[0], -1, (255,255,255), border_thick)  
    return img.astype(np.uint8)

def GauBlur(img):
    return cv2.GaussianBlur(img, (5,5), 10)

def img_mirror(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_mirror = copy.deepcopy(img)
    for i in range(h):
        for j in range(w):
            img_mirror[i, w-1-j] = img[i, j]
    return img_mirror

def plot_hist(score_list, name):
    plt.tick_params()
    plt.hist(np.array(score_list), bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("score")
    # 显示纵轴标签
    plt.ylabel("num")
    # 显示图标题
    plt.title(name)
    plt.savefig(name)

def img_merge(img_list):
    """
    img_list include image readed by cv2.
    """
    img = np.hstack(img_list)
    return img

def white_balance(img):
    img_new = copy.deepcopy(img)
    img_new = np.array(img_new, np.float64)
    aveB = np.mean(img_new[:,:,0])
    aveG = np.mean(img_new[:,:,1])
    aveR = np.mean(img_new[:,:,2])
    aveGray = (aveB + aveG + aveR) / 3
    # 计算增益
    CoefB = aveB / aveGray
    CoefG = aveG / aveGray
    CoefR = aveR / aveGray
    img_new[:,:,0] *= CoefB
    img_new[:,:,1] *= CoefG
    img_new[:,:,2] *= CoefR
    img_new = img_new.clip(0, 255)
    img_new = np.array(img_new, np.uint8)
    return img_new


def Score_analyse(same_score, diff_score, name):
    """
    same_score: list of same scores
    diff_score: list of different scores
    name: picture name, such as *.png
    """
    plt.title("same/different pairs seq score histogram")
    plt.hist(same_score, bins=100, normed = 1, facecolor="red", edgecolor="black", label='same', alpha=0.7, hold = 1)
    plt.hist(diff_score, bins=100, normed = 1, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    plt.xlabel("score")
    plt.ylabel("number")
    plt.legend() # 显示方块
    plt.savefig(name)
    plt.show()


def cos_sim(vector_a, vector_b):
    """
    compute the cos_sim between two vectors
    param: vector_a
    param: vector_b
    return: cos_sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5*cos
    return sim


def get_pairs(img_list):
    """
    返回可迭代对象，里面是每对图片的路径。
    """
    return itertools.combinations(img_list, 2)


if __name__ == "__main__":
    score_list = [math.sin(i) for i in range(10000)]
    plot_hist(score_list, "hist.png")
