import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model

"""
This is a Singleton class which bears the ml model in memory
model is used to extract handshape 
"""
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))


class HandShapeFeatureExtractor:
    __single = None

    @staticmethod
    def get_instance():
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        if HandShapeFeatureExtractor.__single is None:
            real_model = load_model(os.path.join(BASE, 'cnn_model.h5'))
            self.model = real_model
            HandShapeFeatureExtractor.__single = self

        else:
            raise Exception("This Class bears the model, so it is made Singleton")

    # private method to preprocess the image
    @staticmethod
    def __pre_process_input_image(crop):
        try:
            img = cv2.resize(crop, (200, 200))
            img_arr = np.array(img) / 255.0
            img_arr = img_arr.reshape(1, 200, 200, 1)
            return img_arr
        except Exception as e:
            print(str(e))
            raise

    # calculating dimensions for the cropping the specific hand parts
    # Need to change constant 80 based on the video dimensions
    @staticmethod
    def __bound_box(x, y, max_y, max_x):
        y1 = y + 80
        y2 = y - 80
        x1 = x + 80
        x2 = x - 80
        if max_y < y1:
            y1 = max_y
        if y - 80 < 0:
            y2 = 0
        if x + 80 > max_x:
            x1 = max_x
        if x - 80 < 0:
            x2 = 0
        return y1, y2, x1, x2

    def extract_feature(self, image):
        try:
            img_arr = self.__pre_process_input_image(image)
            # input = tf.keras.Input(tensor=image)
            return self.model.predict(img_arr)
        except Exception as e:
            raise


