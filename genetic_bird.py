from FlappyBird import GameState
import numpy as np
import tensorflow as tf


class Genetic(object):
    def __init__(self):
        '''初始化10只鸟遗传'''
        self.num = 10

    def initNetwork(self):
        '''初始化10只鸟的权值与bias'''
        self._n_input_layer = 2
        self._n_hidden_layer_1 = 7
        self._n_output_layer = 2
        self._X = tf.placeholder('float', [None, 2])
        self._Y = tf.placeholder('float', [None])
        self._layer_w_b_1 = []
        self._layer_w_b_output = []
        for i in range(self.num):
            self._layer_w_b_1.append({
                'w':
                tf.Variable(
                    tf.random_normal(
                        [self._n_input_layer, self._n_hidden_layer_1])),
                'b':
                tf.Variable(tf.random_normal([self._n_hidden_layer_1]))
            })
            self._layer_w_b_output.append({
                'w':
                tf.Variable(
                    tf.random_normal(
                        [self._n_hidden_layer_1, self._n_output_layer])),
                'b':
                tf.Variable(tf.random_normal([self._n_output_layer]))
            })

    def predict(self, index, dx, dy):
        '''对某只鸟的状态进行计算出最佳动作 0->不飞  1->飞'''
        layer_1 = tf.add(
            tf.matmul(self._X, self._layer_w_b_1[index]['w']),
            self._layer_w_b_1[index]['b'])
        layer_1 = tf.nn.tanh(layer_1)
        layer_output = tf.add(
            tf.mat_mul(layer_1, self._layer_w_b_output[index]['w']),
            self._layer_w_b_output[index]['b'])
        return np.argmax(layer_output)

    def crossover(self, index1, index2):
        '''对两只鸟的权值进行交叉变化'''
        weight_layer_1_1 = self._layer_w_b_1[index1]
        weight_layer_1_2 = self._layer_w_b_1[index2]
        self._layer_w_b_1[index1] = weight_layer_1_2
        self._layer_w_b_1[index2] = weight_layer_1_1
        weight_layer_output_1 = self._layer_w_b_output[index1]
        weight_layer_output_2 = self._layer_w_b_output[index2]
        self._layer_w_b_output[index1] = weight_layer_output_2
        self._layer_w_b_output[index2] = weight_layer_output_1

    def start(self):
        game = GameState(self.num)
        game.start()
        while True:
            actions = np.random.rand(self.num)
            game.geneticStep(actions)


genetic = Genetic()
genetic.start()
