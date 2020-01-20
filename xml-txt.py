from xml.dom.minidom import parse
import os
import random

labelDict = {
    'person': '0',
    'people': '0',
    'bag': '1'
}


def xml_txt(xmlPath, txtPath):
    xmlList = [x for x in os.listdir(xmlPath) if x.split('.')[-1] == 'xml']
    print(xmlList)
    for xmlName in xmlList:
        annotation = parse(xmlPath + xmlName).documentElement
        objects = annotation.getElementsByTagName('object')
        width = float(annotation.getElementsByTagName('size')[0].getElementsByTagName('width')[0].childNodes[0].data)
        height = float(annotation.getElementsByTagName('size')[0].getElementsByTagName('height')[0].childNodes[0].data)
        txtName = xmlName.split('.')[0] + '.txt'
        with open(txtPath+txtName, 'w') as f:
            for obj in objects:
                label = obj.getElementsByTagName('name')[0].childNodes[0].data
                bndbox = obj.getElementsByTagName('bndbox')[0]
                xmin = float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
                x = (xmin + xmax) / 2 / width
                y = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                content = labelDict[label]
                for e in (x, y, w, h):
                    content += ' ' + str(e)
                f.write(content + '\n')


def gen_dataset(labelPath):
    labelList = [l for l in os.listdir(labelPath) if l.split('.')[-1] == 'txt']
    trainList = random.sample(labelList, int(len(labelList)*0.8))
    validList = list(set(labelList) - set(trainList))
    trainName = '/home/lx/githubProject/PyTorch-YOLOv3/data/custom/train.txt'
    validName = '/home/lx/githubProject/PyTorch-YOLOv3/data/custom/valid.txt'
    with open(trainName, 'w') as f:
        for label in trainList:
            f.write('data/custom/images/' + label.split('.')[0] + '.jpg\n')

    with open(validName, 'w') as f:
        for label in validList:
            f.write('data/custom/images/' + label.split('.')[0] + '.jpg\n')


if __name__ == '__main__':
    # xml_txt('/media/lx/5f5da3ff-2264-4479-82cc-7053cf09f640/车底藏人数据/车底检测数据/labels/',
    #         '/media/lx/5f5da3ff-2264-4479-82cc-7053cf09f640/车底藏人数据/车底检测数据/labels-txt/')
    gen_dataset('/home/lx/githubProject/PyTorch-YOLOv3/data/custom/labels')