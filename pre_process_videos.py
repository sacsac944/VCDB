import tensorflow as tf
import numpy as np


def preprocess_frame_inception_v3(frame: np.ndarray):
    frame = tf.image.resize(frame, (299, 299))
    frame = tf.keras.applications.inception_v3.preprocess_input(frame)
    return frame


def preprocess_frame_resnet_50(frame: np.ndarray):
    frame = tf.image.resize(frame, (224, 224))
    frame = tf.keras.applications.resnet50.preprocess_input(frame)
    return frame


def preprocess_frame_vgg_19(frame: np.ndarray):
    frame = tf.image.resize(frame, (224, 224))
    frame = tf.keras.applications.vgg19.preprocess_input(frame)
    return frame
