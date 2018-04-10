from FlappyBird import GameState
import tensorflow as tf
import numpy as np


class DQN(object):
    def __init__(self):
        pass

    def initNetwork(self):
        '''输入图像为[80,80,1]'''
        weights_conv1=tf.Variable()