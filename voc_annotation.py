import xml.etree.ElementTree as ET
import os
from os import getcwd

#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=['train', 'val', 'test']

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["kangaroo"]

def convert_annotation(image_set,image_id, list_file):
    in_file = open('kangaroo/%s/annots/%s.xml'%(image_set,image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    #if not os.path.exists('kangaroo/labels/'):
    #    os.makedirs('kangaroo/labels/')
    #image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    #for image_id in image_ids:
    for s in range(1,184):
        numstr = str(s)                                                    
        image_id = numstr.zfill(5)                                                       
        print (image_id)
        if not os.path.exists('kangaroo/%s/images/%s.jpg'% (image_set,image_id)):
            continue
        list_file.write('%s/kangaroo/images/%s.jpg'%(wd, image_id))
        #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(image_set,image_id, list_file)
        list_file.write('\n')
    list_file.close()

