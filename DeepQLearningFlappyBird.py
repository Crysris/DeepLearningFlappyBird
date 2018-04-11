from FlappyBird import GameState
import tensorflow as tf
import numpy as np
from collections import deque
import cv2
import random

BATCH_SIZE = 64
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 2000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000


class DQN(object):
    def __init__(self):
        self._x = tf.placeholder('float', [None, 80, 80, 1])
        self._y = tf.placeholder('float', [None])
        self._a = tf.placeholder('float', [None, 2])

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(
            x, w, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool(self, x, stride):
        return tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, stride, stride, 1],
            padding="SAME")

    def initNetwork(self):
        '''输入图像为[80,80,1]'''
        # 第一层卷积   filter=[2,2,1,20] stride=2  [80,80,1]==>[40,40,20]
        weights_conv1 = tf.Variable(
            tf.truncated_normal([2, 2, 1, 20], stddev=0.1))
        bias_conv1 = tf.Variable(tf.truncated_normal([20], stddev=0.1))
        # 第一层池化 filter=[2,2] stride=2     [40,40,20]==>[20,20,20]

        # 第二层卷积 filter=[2,2,20,40] stride=2   [20,20,20]==>[10,10,40]
        weights_conv2 = tf.Variable(
            tf.truncated_normal([2, 2, 20, 40], stddev=0.1))
        bias_conv2 = tf.Variable(tf.truncated_normal([40], stddev=0.1))
        # 第二层池化 filter=[2,2] stride=2 [10,10,40]==>[5,5,40]

        # 第三层卷积 filter=[2,2,40,60] stride=2 [5,5,40]==>[3,3,60]
        # 第三层池化 filter[2,2] stride=2 [3,3,60]==>[2,2,60]
        weights_conv3 = tf.Variable(
            tf.truncated_normal([2, 2, 40, 60], stddev=0.1))
        bias_conv3 = tf.Variable(tf.truncated_normal([60], stddev=0.1))

        # [2,2,60]==>[1,240]

        # 全连接层1 [240,80]          [1,240]==>[1,80]
        weights_fc1 = tf.Variable(tf.truncated_normal([240, 80], stddev=0.1))
        bias_fc1 = tf.Variable(tf.truncated_normal([80], stddev=0.1))

        # 全连接层2 [80,2]           [1,80]==>[1,2]
        weights_fc2 = tf.Variable(tf.truncated_normal([80, 2], stddev=0.1))
        bias_fc2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))

        h_conv1 = tf.nn.tanh(
            tf.add(self.conv2d(self._x, weights_conv1, 2), bias_conv1))
        h_pool1 = self.max_pool(h_conv1, 2)

        h_conv2 = tf.nn.tanh(
            tf.add(self.conv2d(h_pool1, weights_conv2, 2), bias_conv2))
        h_pool2 = self.max_pool(h_conv2, 2)
        h_conv3 = tf.nn.tanh(
            tf.add(self.conv2d(h_pool2, weights_conv3, 2), bias_conv3))
        h_pool3 = self.max_pool(h_conv3, 2)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 240])

        h_fc1 = tf.nn.tanh(
            tf.add(tf.matmul(h_pool3_flat, weights_fc1), bias_fc1))

        readout = tf.add(tf.matmul(h_fc1, weights_fc2), bias_fc2)
        return readout

    def trainNetwork(self, readout, sess):
        readout_t = tf.reduce_mean(tf.multiply(readout, self._a))
        cost = tf.reduce_mean(tf.square(self._y - readout_t))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        game = GameState()
        game.start()
        action = np.array([0, 1])
        action_idx = 0
        img_data, reward, survived = game.qearningStep(action)
        img_data = cv2.cvtColor(
            cv2.resize(img_data, (80, 80)), cv2.COLOR_BGR2GRAY)

        ret, img_data = cv2.threshold(img_data, 1, 255, cv2.THRESH_BINARY)
        img_data = np.reshape(img_data, (80, 80, 1))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        D = deque()
        checkpoint = tf.train.get_checkpoint_state('saved_network_qlearning')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('network loaded successfully!',
                  checkpoint.model_checkpoint_path)
        else:
            print('network loaded failed!')

        step = 0
        life = 0
        maxLife = 0
        epsilon = INITIAL_EPSILON
        while True:
            readout_t = sess.run(readout, feed_dict={self._x: [img_data]})
            action = np.zeros(2)
            if np.random.random() <= epsilon:
                action_idx = np.random.randint(2)
                action[action_idx] = 1
            else:
                action_idx = np.argmax(readout_t)
                action[action_idx] = 1

            img_data1, reward, survived = game.qearningStep(action)
            if survived:
                life += 1
            else:
                life = 0
            maxLife = max(maxLife, life)

            img_data1 = cv2.cvtColor(
                cv2.resize(img_data1, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, img_data1 = cv2.threshold(img_data1, 1, 255,
                                           cv2.THRESH_BINARY)
            img_data1 = np.reshape(img_data1, (80, 80, 1))
            D.append([img_data, action, reward, survived, img_data1])

            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if step > OBSERVE:
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                batch = random.sample(D, BATCH_SIZE)

                # 获得训练样本中的数据
                img_batch = [d[0] for d in batch]
                a_batch = [d[1] for d in batch]
                r_batch = [d[2] for d in batch]
                img1_batch = [d[4] for d in batch]

                y_batch = []
                readout1_batch = sess.run(
                    readout, feed_dict={self._x: img1_batch})

                for i in range(len(batch)):
                    survived = batch[i][3]
                    if survived:
                        y_batch.append(
                            r_batch[i] + GAMMA * np.max(readout1_batch[i]))
                    else:
                        y_batch.append(r_batch[i])

                # 梯度下降
                train_step.run(feed_dict={
                    self._x: img_batch,
                    self._a: a_batch,
                    self._y: y_batch
                })

            img_data = img_data1
            step += 1

            if step % 500000 == 0:
                saver.save(
                    sess, 'saved_network_qlearning/dql', global_step=step)

            state = ''
            if step < OBSERVE:
                state = 'observe'
            elif step < EXPLORE:
                state = 'explore'
            else:
                state = 'train'

            print('step', step, '/state', state, '/epsilon', epsilon,
                  '/action', action_idx, '/reward', reward, '/life', life,
                  '/maxLife', maxLife)

    def start(self):
        sess = tf.InteractiveSession()
        readout = self.initNetwork()
        self.trainNetwork(readout, sess)


game = DQN()
game.start()