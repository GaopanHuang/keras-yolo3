#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
get mAP of a YOLO_v3 style detection model on valuation images.
"""
import os
#gpu_id = '2,3'
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
#config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

import colorsys
import io
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/trained_weights_final_raccoon.h5' # model path or trained weights path
        #self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/raccoon_classes.txt'
        #self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45 #iou of NMS, using for detection
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
        
    def _compute_overlap(self, a, b):
        """
        Args
            a: (N, 4) ndarray of float
            b: (K, 4) ndarray of float
        Returns
            overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    
        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
    
        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)
    
        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    
        ua = np.maximum(ua, np.finfo(float).eps)
    
        intersection = iw * ih
    
        return intersection / ua

    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
    
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
    
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _get_map(self, all_detections, all_annotations, 
                iou_threshold=0.5):
        '''Calculate mAP with a given detections.
        # Arguments
            all_detections  : a given detections, n*num_class*d*5, where n is number of images, 
                              num_class is the number of object class, d is the number of detected boundingboxs,
                              5 is the rect of BB and score.
            all_annotations : a given annotations, n*num_class*d*4, like detections.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        '''
                
        average_precisions = {}
    
        # process detections and annotations
        for label in range(len(self.class_names)):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0
    
            for i in range(len(all_detections)):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += len(annotations)
                detected_annotations = []
    
                for d in detections:
                    scores = np.append(scores, d[4])
    
                    if len(annotations) == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue
    
                    overlaps            = self._compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]
    
                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
    
            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue
    
            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]
    
            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)
    
            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    
            # compute average precision
            average_precision  = self._compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations
    
        return average_precisions    

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    
    def close_session(self):
        self.sess.close()
        
    def evaluate(self,
                 annotation_path,
                 iou_threshold=0.5,
                 save_img = False):
        '''Evaluate a given dataset using a given model.
        # Arguments
            model           : The model to evaluate.
            annotation_path : each row in imgset_lable file like this:
                              imgpath x_min1,y_min1,x_max1,y_max1,0 x_min2,y_min2,x_max2,y_max2,2......
            iou_threshold   : The threshold used to consider when a detection is positive or negative, 
                              using for mAP, not the same as iou of NMS for detections.
        # Returns
            A dict mapping class names to mAP scores.
        '''
        with open(annotation_path) as f:
            annotation_lines = f.readlines()
        all_detections = [[None for i in range(len(self.class_names))] for j in range(len(annotation_lines))]
        all_annotations = [[None for i in range(len(self.class_names))] for j in range(len(annotation_lines))]

        start = timer()

        for i in range(len(annotation_lines)):
            line = annotation_lines[i].split()
            image = Image.open(line[0])
            boxes = np.array([np.array(list(map(int,boxes.split(',')))) for boxes in line[1:]])

            for label in range(len(self.class_names)):
                all_annotations[i][label] = boxes[boxes[:, -1] == label, :-1]
            
            if self.model_image_size != (None, None):
                assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            else:
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                boxed_image = letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            out_boxes2 = out_boxes.copy()
            for j in range(len(out_boxes2)):
                box = out_boxes2[j]
                top, left, bottom, right = box
                out_boxes2[j][1] = max(0, np.floor(top + 0.5).astype('int32'))
                out_boxes2[j][0] = max(0, np.floor(left + 0.5).astype('int32'))
                out_boxes2[j][3] = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                out_boxes2[j][2] = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            image_detections = np.concatenate([out_boxes2, np.expand_dims(out_scores, axis=1), np.expand_dims(out_classes, axis=1)], axis=1)
            for label in range(len(self.class_names)):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

            if save_img:
                font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300
        
                for m, c in reversed(list(enumerate(out_classes))):
                    predicted_class = self.class_names[c]
                    box = out_boxes[m]
                    score = out_scores[m]
        
                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
        
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))
        
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])
        
                    # My kingdom for a good redistributable image drawing library.
                    for n in range(thickness):
                        draw.rectangle(
                            [left + n, top + n, right - n, bottom - n],
                            outline=self.colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw
                
                image.save('results/val/val_%d.jpg'%(i))

        end = timer()
        print(end - start)
        return self._get_map(all_detections, all_annotations, iou_threshold=iou_threshold)

def evaluate_map(yolo):
    print (yolo.evaluate('raccoon_val.txt',0.5,True))
    yolo.close_session()



if __name__ == '__main__':
    evaluate_map(YOLO())
