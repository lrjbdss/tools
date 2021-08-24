from xml.dom.minidom import parse
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageTk, ImageFont
import cv2
import random
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import collections


label = None
resize_max = 540


def del_img():
    img_dir = './images'
    label_dir = './labels'
    res_dir = './det_result'
    imgs = os.listdir(res_dir)
    for image in [img for img in os.listdir(img_dir) if img.endswith('.jpg')]:
        if image not in imgs:
            label_path = os.path.join(label_dir, image[:-4]+'.txt')
            img_path = os.path.join(img_dir, image)
            os.remove(label_path)
            os.remove(img_path)


def create_set():
    txt_dir = './labels/'
    saveBasePath = "./"
    total_txt = os.listdir(txt_dir)
    random.shuffle(total_txt)
    train_lst = total_txt[:28000]
    val_lst = total_txt[28000:]

    ftrain = open(os.path.join(saveBasePath, 'imgsets/train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'imgsets/val.txt'), 'w')
    for txt in train_lst:
        name = txt[:-4]+'\n'
        ftrain.write(name)
    for txt in val_lst:
        name = txt[:-4]+'\n'
        fval.write(name)
    ftrain.close()
    fval.close()


def cut_obj():
    label_dir = './labels'
    img_dir = './images'
    obj_save_dir = './head_img'
    if os.path.exists(obj_save_dir):
        shutil.rmtree(obj_save_dir)
    os.mkdir(obj_save_dir)
    count = 0
    pbar = tqdm(sorted(os.listdir(img_dir)), desc='obj count: ')
    for img_name in pbar:
        if img_name.endswith('.jpg'):
            label_path = os.path.join(label_dir, img_name[:-3]+'txt')
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            for line in open(label_path, 'r'):
                _, x, y, w, h = [eval(d) for d in line.strip().split(' ')]
                x1 = int(max((x - w/2)*img_w, 0))
                x2 = int(min((x + w/2)*img_w, img_w))
                y1 = int(max((y - h/2)*img_h, 0))
                y2 = int(min((y + h/2)*img_h, img_h))
                if max(x2-x1, y2-y1) >= 48:
                    obj = img[y1:y2, x1:x2]
                    # print('x1,x2,y1,y2:(%d, %d, %d, %d)' % (x1, x2, y1, y2))
                    save_path = os.path.join(obj_save_dir, '%d.jpg' % count)
                    cv2.imwrite(save_path, obj)
                    count += 1
                    pbar.desc = 'obj count: %5d' % count


def get_xmllist():
    xmllist = [i
               for i in glob.glob("{}/*.xml".format(xml_path))]
    print("xml list len: ", len(xmllist))
    return xmllist


def to_txt(xmls, txt_save_dir):
    random.shuffle(xmls)
    if os.path.exists(txt_save_dir):
        shutil.rmtree(txt_save_dir)
    os.mkdir(txt_save_dir)
    cls_dict = {'baby_face': 0,
                'other_face': 1}
    for xmlfile in xmls:
        print(xmlfile)
        pic_name = xmlfile.split('/')[-1][:-3] + 'jpg'
        pic_path = os.path.join(jpg_path, pic_name)
        if not os.path.exists(pic_path):
            continue
        dom = parse(xmlfile)
        annotation = dom.documentElement

        size = annotation.getElementsByTagName('size')[0]
        W = int(size.getElementsByTagName('width')[0].childNodes[0].data)
        H = int(size.getElementsByTagName('height')[0].childNodes[0].data)
        if W*H == 0:
            img = cv2.imread(pic_path)
            H, W, _ = img.shape
        objects = annotation.getElementsByTagName('object')
        txt_name = xmlfile.split('/')[-1][:-3] + 'txt'
        txt_path = os.path.join(txt_save_dir, txt_name)
        with open(txt_path, 'w') as f:
            for obj in objects:
                bndbox = obj.getElementsByTagName('bndbox')[0]
                name = obj.getElementsByTagName('name')[0].childNodes[0].data
                xmin = int(bndbox.getElementsByTagName(
                    'xmin')[0].childNodes[0].data)
                ymin = int(bndbox.getElementsByTagName(
                    'ymin')[0].childNodes[0].data)
                xmax = int(bndbox.getElementsByTagName(
                    'xmax')[0].childNodes[0].data)
                ymax = int(bndbox.getElementsByTagName(
                    'ymax')[0].childNodes[0].data)
                w = (xmax-xmin) / W
                h = (ymax-ymin) / H
                xc = xmin / W + w/2
                yc = ymin / H + h/2
                line = ' '.join([str(i)
                                 for i in [cls_dict[name], xc, yc, w, h]]) + '\n'
                f.write(line)


def resize_img(img_dir):
    imgs = os.listdir(img_dir)
    print("pic num: %d" % len(imgs))
    for i, img_name in enumerate(imgs):
        if img_name.endswith('.jpg'):
            print('%d: %s' % (i, img_name))
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            if min(w, h) == resize_max:
                continue
            scale = resize_max / h if h < w else resize_max / w
            img = cv2.resize(img, (int(scale*w), int(scale*h)))
            dst = os.path.join(src, "images")
            if not os.path.exists(dst):
                os.makedirs(dst)

            cv2.imwrite(os.path.join(dst, img_name), img)


def yolov5_dataset(img_dir, label_dir):
    dataset_dir = './yolov5_dataset'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)
    img_train = './yolov5_dataset/images/train/'
    img_val = './yolov5_dataset/images/val/'
    label_train = './yolov5_dataset/labels/train/'
    label_val = './yolov5_dataset/labels/val/'
    os.mkdir('./yolov5_dataset/images')
    os.mkdir('./yolov5_dataset/labels')
    os.mkdir(img_train)
    os.mkdir(img_val)
    os.mkdir(label_train)
    os.mkdir(label_val)
    sets = {'train': './ImageSets/Main/train.txt',
            'val': './ImageSets/Main/test.txt'}
    for k, v in sets.items():
        for line in open(v, 'r'):
            name = line.strip()
            img_name = name+'.jpg'
            txt_name = name+'.txt'
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(label_dir, txt_name)
            if k == 'train':
                save_img_path = os.path.join(img_train, img_name)
                save_txt_name = os.path.join(label_train, txt_name)
            else:
                save_img_path = os.path.join(img_val, img_name)
                save_txt_name = os.path.join(label_val, txt_name)
            shutil.copy(img_path, save_img_path)
            shutil.copy(txt_path, save_txt_name)


def cal_anchor(txt_dir):
    txt_list = [os.path.join(txt_dir, txt) for txt in os.listdir(txt_dir)]
    whs = []
    for txt_path in txt_list:
        for line in open(txt_path, 'r'):
            whs.append([float(d) for d in line.strip().split(' ')[-2:]])
    whs = np.array(whs) * np.array([512, 288])
    # whs = whs.reshape((-1, 1))
    print(len(whs))
    n_clusters = 9
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(whs)
    anchors = kmeans.cluster_centers_
    anchors = np.array(sorted(anchors, key=lambda anchor: anchor[0]))
    print(anchors)
    # result = []
    # for i in range(n_clusters):
    #     result.append([anchors[i][0], (kmeans.labels_ == i).sum()])

    # result.sort(key=lambda t: t[0])
    # for r in result:
    #     print(r)


def show_rect():
    label_dir = '/home/inomjon/work/data/baby_head/labels'
    img_dir = '/home/inomjon/work/data/baby_head/JPEGImages'
    obj_save_dir = '/home/inomjon/work/data/baby_head/show_rect'
    if os.path.exists(obj_save_dir):
        shutil.rmtree(obj_save_dir)
    os.mkdir(obj_save_dir)
    for label_name in sorted(os.listdir(label_dir)):
        if label_name.endswith('.txt'):
            label_path = os.path.join(label_dir, label_name)
            img_path = os.path.join(img_dir, label_name[:-3]+'jpg')
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            for line in open(label_path, 'r'):
                _, x, y, w, h = [eval(d) for d in line.strip().split(' ')]
                x1 = int(max((x - w/2)*img_w, 0))
                x2 = int(min((x + w/2)*img_w, img_w))
                y1 = int(max((y - h/2)*img_h, 0))
                y2 = int(min((y + h/2)*img_h, img_h))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                save_path = os.path.join(obj_save_dir, label_name[:-3]+'jpg')
                cv2.imwrite(save_path, img)


def save_diff_img():
    ori_img_dir = '/home/inomjon/work/data/baby_0218/auto_label/baby_0218_fixed_label/images'
    saved_img_dir = '/home/inomjon/work/data/baby_0218/images'
    diff_save_dir = '/home/inomjon/work/data/baby_0218/mis_img'
    ori_img_list = os.listdir(ori_img_dir)
    saved_img_list = os.listdir(saved_img_dir)
    diff_imgs = [name for name in ori_img_list if name not in saved_img_list]
    for name in diff_imgs:
        img_path = os.path.join(ori_img_dir, name)
        save_img_path = os.path.join(diff_save_dir, name)
        shutil.copy(img_path, save_img_path)


def err_img_set():
    err_img_dir = '/home/inomjon/mnt/err_imgs'
    err_label_dir = '/home/inomjon/mnt/err_labels'
    img_dir = '/home/inomjon/mnt/images'
    label_dir = '/home/inomjon/mnt/labels'
    imgset_dir = '/home/inomjon/mnt/imgsets'

    # 清理旧软连接
    [os.remove(name)
     for name in filter(os.path.islink, glob.glob(img_dir+'/*.jpg'))]
    [os.remove(name) for name in filter(
        os.path.islink, glob.glob(label_dir+'/*.txt'))]

    id_list = [name.split('.')[0] for name in os.listdir(err_img_dir)]
    ferr = open(os.path.join(imgset_dir, 'err.txt', ), 'w')
    for img_name in id_list:
        line = img_name+'\n'
        ferr.write(line)
        # 生成标签文件
        label_path = os.path.join(err_label_dir, img_name+'.txt')
        if not os.path.exists(label_path):
            os.mknod(label_path)
        # 在images和labels中创建软连接
        sym_img_path = os.path.join(img_dir, img_name+'.jpg')
        os.symlink('/data/longxiang/baby_0419/err_imgs/%s.jpg' %
                   img_name, sym_img_path)
        sym_label_path = os.path.join(label_dir, img_name+'.txt')
        os.symlink('../err_labels/%s.txt' % img_name, sym_label_path)

    ferr.close()


def video2img():
    video_id = 45  # 录像id
    skip = 20  # 采样最小间隔
    img_num = 5  # 一个录像取样数量
    video_path = '/home/inomjon/mnt/test/%d.mp4' % video_id
    save_path = '/home/inomjon/mnt/err_imgs'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    img_id = 0
    while ret:
        cv2.imshow('%d.mp4' % video_id, frame)
        ch = cv2.waitKey(0) & 0xFF
        if ch == ord('s'):
            img = cv2.resize(frame, (640, 360))
            cv2.imwrite(os.path.join(save_path, '0%d_%d.jpg' %
                                     (video_id, img_id)), img)
            img_id += 1
            if img_id == img_num:
                break
            for _ in range(skip):
                ret, frame = cap.read()
        else:
            ret, frame = cap.read()
    cap.release()


def yuv2jpg():
    root_dir = './baby_data'
    save_dir = './images'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    device_num = 0
    for root, dirs, files in os.walk(root_dir):
        for img_name in files:
            if img_name.endswith('.yuv'):
                print(os.path.join(root, img_name))
                yuv_path = os.path.join(root, img_name)
                yuv = np.fromfile(yuv_path, dtype=np.uint8)
                if yuv.size == 345600:
                    yuv = yuv.reshape(540, 640)
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                    save_path = os.path.join(
                        save_dir, '%d_%s' % (device_num, img_name[5:-3] + 'jpg'))
                    cv2.imwrite(save_path, bgr, [
                                int(cv2.IMWRITE_JPEG_QUALITY), 100])
        device_num += 1


def randByID():
    res_dir = './auto_label/det_result'
    img_dir = './auto_label/images'
    label_dir = './auto_label/labels'
    save_img_dir = './images'
    save_label_dir = './labels'
    save_res_dir = './det_result'
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    os.mkdir(save_img_dir)
    if os.path.exists(save_label_dir):
        shutil.rmtree(save_label_dir)
    os.mkdir(save_label_dir)
    if os.path.exists(save_res_dir):
        shutil.rmtree(save_res_dir)
    os.mkdir(save_res_dir)

    savePerID = 20
    imgs = [name[:-4] for name in os.listdir(res_dir) if name.endswith('.jpg')]
    id_dct = collections.defaultdict(list)
    for name in imgs:
        key = name.split('_')[0]
        id_dct[key].append(name)
    for key in id_dct:
        if len(id_dct[key]) > savePerID:
            id_dct[key] = random.sample(id_dct[key], savePerID)

    for key in id_dct:
        for name in id_dct[key]:
            save_img_path = os.path.join(save_img_dir, name+'.jpg')
            save_label_path = os.path.join(save_label_dir, name+'.txt')
            save_res_path = os.path.join(save_res_dir, name+'.jpg')
            img_path = os.path.join(img_dir, name+'.jpg')
            label_path = os.path.join(label_dir, name+'.txt')
            res_path = os.path.join(res_dir, name+'.jpg')
            shutil.copy(img_path, save_img_path)
            shutil.copy(label_path, save_label_path)
            shutil.copy(res_path, save_res_path)


def copy_labels():
    root = './labels'
    mxg = 'mxg_labels'
    zhx = 'zhx_labels'
    lst = os.listdir(root)
    lst.sort(key=lambda name: [int(num) for num in name[:-4].split('_')])
    idx = 0
    for i, name in enumerate(lst):
        if '1300_' in name:
            idx = i
            break
    for i in range(idx):
        name = lst[i]
        src = os.path.join(mxg, name)
        dst = os.path.join(root, name)
        shutil.copy(src, dst)
    for i in range(idx, len(lst)):
        name = lst[i]
        src = os.path.join(zhx, name)
        dst = os.path.join(root, name)
        shutil.copy(src, dst)


def check_label():
    print("请输入目录：")
    root = input()
    while(not os.path.exists(root)):
        print('目录不存在，请重新输入')
        root = input()

    print("请选择：\n   0:顺序\n   1:倒序\n")
    flag = int(input())
    lst = os.listdir(root)
    lst.sort(key=lambda name: [int(num)
                               for num in name[:-4].split('_')], reverse=bool(flag))
    idx = 0
    n = len(lst)
    print("请输入要跳转的设备ID号")
    device_id = input()
    while(True):
        try:
            dev_id = int(device_id)
            assert(dev_id < n)
        except:
            print('输入有误，请重新输入')
            device_id = input()
        else:
            break

    for i in range(idx+1, n):
        name = lst[i]
        if device_id == name.split('_')[0]:
            idx = i
            break

    while(idx < n):
        name = lst[idx]
        print(name)
        src = os.path.join(root, name)
        for line in open(src, 'r'):
            label = line.split(' ')[0]
            if len(label) != 4:
                print(label, "标签格式有误，请修正，然后输入要跳转的设备ID号，直接回车不跳转")
                device_id = input()

                if device_id != '':
                    while(True):
                        try:
                            dev_id = int(device_id)
                            assert(dev_id < n)
                        except:
                            print('输入有误，请重新输入')
                            device_id = input()
                        else:
                            break
                    for i in range(idx+1, n):
                        name = lst[i]
                        if device_id == name.split('_')[0]:
                            idx = i - 1
                            break
                else:
                    break
            break
        idx += 1


def sortByLoss():
    txt_path = './imgsets/best9.txt'
    df = pd.read_csv(txt_path, sep='\s+')
    df['train'] = (df.index > 2172).astype(int)

    # df.sort_values('total', ascending=False).set_index(
    #     'img').to_excel('imgsets/loss.xlsx')
    # df.sort_values('hm', ascending=False).loc[:, 'img'].to_csv(
    #     'imgsets/hm_loss.txt', '\n', index=False, header=False)
    # df.sort_values('wh', ascending=False).loc[:, 'img'].to_csv(
    #     'imgsets/wh_loss.txt', '\n', index=False, header=False)
    # df.sort_values('ab', ascending=False).loc[:, 'img'].to_csv(
    #     'imgsets/ab_loss.txt', '\n', index=False, header=False)
    # df.sort_values('eye', ascending=False).loc[:, 'img'].to_csv(
    #     'imgsets/eye_loss.txt', '\n', index=False, header=False)
    # df.sort_values('twd', ascending=False).loc[:, 'img'].to_csv(
    #     'imgsets/twd_loss.txt', '\n', index=False, header=False)

    idx = (df['total'] > 0.18) | ((df['wh'] == 0)
                                  & (df['total'] > 0.06)) & df['train']
    df.loc[idx, 'img'].to_csv(
        'imgsets/train_loss.txt', '\n', index=False, header=False)


if __name__ == '__main__':
    # to_txt(get_xmllist(), './labels')
    # yolov5_dataset('./JPEGImages', './labels')
    # cal_anchor("./labels")
    # create_set()
    # del_img()
    # cut_obj()
    # show_rect()
    # save_diff_img()
    err_img_set()
    # video2img()
    # yuv2jpg()
    # randByID()
    # copy_labels()
    # check_label()
    # sortByLoss()
