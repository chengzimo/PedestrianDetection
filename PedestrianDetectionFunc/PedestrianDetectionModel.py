#encoding: utf-8
import numpy as np
from PedestrianDetectionFunc.Utils.utils import letterbox_image
from PedestrianDetectionFunc.Model.model import yolo_eval
import os
from keras import backend as K
from keras.models import load_model

class PedestrianModel(object):
    def __init__(self):
        self.input_shape = (416, 416)
        self.model_path = './PedestrianDetectionFunc/Data/model.h5'
        self.anchors_path = './PedestrianDetectionFunc/Data/yolo_anchors.txt'
        self.classes_path = './PedestrianDetectionFunc/Data/coco_classes.txt'
        self.class_names = self.get_classes()
        self.anchors = self.get_anchors()
        self.score = 0.3
        self.iou = 0.5
        self.boxes, self.scores, self.classes = None, None, None
        self.input_image_shape = K.placeholder(shape=(2,))
        self.model = None

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)

    def get_classes(self):
        '''loads the classes'''
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        '''loads the anchors from a file'''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def Preprocess(self, image):
        if self.input_shape != (None, None):
            assert self.input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.input_shape[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.input_shape)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data

    def generate(self):
        self.boxes, self.scores, self.classes = yolo_eval(self.model.output,
                                                          self.anchors,
                                                          len(self.class_names),
                                                          self.input_image_shape,
                                                          score_threshold=self.score,
                                                          iou_threshold=self.iou)










