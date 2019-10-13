#encoding: utf-8
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from keras import backend as K
from PedestrianDetectionFunc.PedestrianDetectionModel import PedestrianModel

class Pedestrian(object):
    def __init__(self):
        K.clear_session()
        self.GPU_config = tf.ConfigProto()
        self.GPU_config.gpu_options.per_process_gpu_memory_fraction = 0.10

    def restore(self, model_path='./PedestrianDetectionFunc/Data/model.h5'):
        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.GPU_config, graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.PedestrianModel = PedestrianModel()
                self.PedestrianModel.load_model(model_path)
                self.PedestrianModel.generate()

    def MakePedestrianDetectionFunc(self, mImage):
        image = self.PedestrianModel.Preprocess(mImage)
        w, h = mImage.size[1], mImage.size[0]
        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.PedestrianModel.boxes, self.PedestrianModel.scores, self.PedestrianModel.classes],
            feed_dict={
                self.PedestrianModel.model.input: image,
                self.PedestrianModel.input_image_shape: [w, h],
            })
        end = timer()
        t = end - start
        out_boxes = out_boxes[out_classes == 0]  # Pedestrian id of coco is 0
        out_scores = out_scores[out_classes == 0]
        out_classes = out_classes[out_classes[:] == 0]

        ResponseStr = {'result': 'success'}
        ResponseStr["box"] = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.PedestrianModel.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(mImage.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(mImage.size[0], np.floor(right + 0.5).astype('int32'))

            res_box = "{} {:.2f} {} {} {} {}".format(predicted_class, score, left, top, right, bottom)
            ResponseStr["box"].append(res_box)
        ResponseStr["time"] = "{:.2f}ms".format(t*100)
        return 1, ResponseStr

def GetPedestrianRes(mImage, Pedestrianclass):
    ResFlag, ResStr = Pedestrianclass.MakePedestrianDetectionFunc(mImage)
    return ResFlag, ResStr




