# -*- encoding:utf-8
import tensorflow as tf
from FlappyBird import GameState
import random
import numpy as np
from collections import deque

INITIAL_EPSILON = 0.0001
FINAL_EPSILON = 0.0001

REPLAY_MEMORY = 50000
GAMMA = 0.001
BATCH_SIZE = 32  # batch大小
OBSERVE = 100  # 观察游戏次数
EXPLORE = 2000000


class QLearning(object):
    def __init__(self):
        self._n_input_layer = 2
        self._n_hidden_layer_1 = 10
        self._n_hidden_layer_2 = 6
        self._n_output_layer = 2
        self._X = tf.placeholder('float', [None, 2])
        self._Y = tf.placeholder('float', [None])
        self._a = tf.placeholder('float', [None, 2])

    def neural_network(self):
        layer_1_w_b = {
            'w':
            tf.Variable(
                tf.random_normal([self._n_input_layer,
                                  self._n_hidden_layer_1])),
            'b':
            tf.Variable(tf.random_normal([self._n_hidden_layer_1]))
        }
        layer_2_w_b = {
            'w':
            tf.Variable(
                tf.random_normal(
                    [self._n_hidden_layer_1, self._n_hidden_layer_2])),
            'b':
            tf.Variable(tf.random_normal([self._n_hidden_layer_2]))
        }
        layer_output_w_b = {
            'w':
            tf.Variable(
                tf.random_normal(
                    [self._n_hidden_layer_2, self._n_output_layer])),
            'b':
            tf.Variable(tf.random_normal([self._n_output_layer]))
        }

        # w*x+b
        layer_1 = tf.add(
            tf.matmul(self._X, layer_1_w_b['w']), layer_1_w_b['b'])
        layer_1 = tf.nn.tanh(layer_1)
        layer_2 = tf.add(
            tf.matmul(layer_1, layer_2_w_b['w']), layer_2_w_b['b'])
        layer_2 = tf.nn.tanh(layer_2)
        layer_output = tf.add(
            tf.matmul(layer_2, layer_output_w_b['w']), layer_output_w_b['b'])

        return layer_output

    def trainNetwork(self, sess, readout):
        # 定义costFunction
        readout_action = tf.reduce_sum(
            tf.multiply(readout, self._X), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self._Y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # 启动FlappyBird游戏
        game = GameState()

        # 用队列储存观察的游戏记录
        D = deque()

        action = 1
        # 得到第一次飞行后的dx，dy,存活状态
        survived, s_t, r_0 = game.step(action)
        epsilon = INITIAL_EPSILON
        t = 0
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state('saved_network')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("network loaded succeccfully!")
        else:
            print("network loaded failed!")
        while True:
            readout_t = readout.eval(feed_dict={self._X: [s_t]})

            # 选择action
            if random.random() <= epsilon:
                # 随机选择一个action
                action = np.random.randint(0, 1)
            else:
                action = np.argmax(readout_t)

            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # 执行游戏
            survived, s_t1, r_1 = game.step(action)

            # 把状态转移存储在队列中
            # s_t : 状态s的dx，dy
            # r_t ：执行action后的奖励
            # s_t1： 状态s1的dx，dy
            D.append((s_t, action, r_1, s_t1, survived))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # 在观察结束后开始训练
            if t > OBSERVE:
                minBatch = random.sample(D, BATCH_SIZE)
                # 在batch中取出训练数据
                s_t_batch = [d[0] for d in minBatch]
                action_batch = [d[1] for d in minBatch]
                r_batch = [d[2] for d in minBatch]
                s_t1_batch = [d[3] for d in minBatch]
                y_batch = []

                readout_t1_batch = readout.eval(
                    feed_dict={self._X: s_t1_batch})

                for i in range(len(minBatch)):
                    survived = minBatch[i][4]
                    # 当鸟存活时才更新网络
                    if survived:
                        y_batch.append(
                            r_batch[i] + GAMMA * np.max(readout_t1_batch[i]))
                    else:
                        y_batch.append(r_batch[i])

                # 梯度下降
                train_step.run(feed_dict={
                    self._Y: y_batch,
                    self._a: action_batch,
                    self._X: s_t_batch
                })

            s_t = s_t1
            t += 1
            # 每迭代10000次存储神经网络数据
            if t % 1000 == 0:
                saver.save(sess, 'saved_network/bird', global_step=t)

            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", r_1, \
                "/ Q_MAX %e"% np.max(readout_t))

    def rl(self):
        xRange = 600
        yRange = 1200
        self.qaVal = np.load('250000-qaval.npy')

        game = GameState()
        action = 1
        survived, bird_data, reward = game.step(action)
        epsilon = INITIAL_EPSILON
        t = 0
        while True:
            if t % 200000 == 0:
                np.save(str(t) + '-qaval.npy', self.qaVal)
            if t >= 2000000:
                break
            s = ''
            # 选择action
            if random.random() <= epsilon:
                # 随机选择一个action
                action = np.random.randint(0, 1)
                s = 'random'
            else:
                action = np.argmax(
                    self.qaVal[bird_data[0]][bird_data[1] + 600])
                s = 'QVal'

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            survived, bird_data1, reward = game.step(action)

            if survived:
                val = self.qaVal[bird_data[0]][bird_data[1] + 600][action]
                self.qaVal[bird_data[0]][bird_data[1] + 600][
                    action] = val + 0.7 * (
                        max(self.qaVal[bird_data1[0]][bird_data1[1]
                                                      + 600] + reward - val))
            else:
                self.qaVal[bird_data[0]][bird_data[1] + 600][action] = reward

            print(t, s, action, reward)
            t += 1

    def play(self):
        self.rl()
        '''sess = tf.InteractiveSession()
        layer_output = self.neural_network()
        self.trainNetwork(sess, layer_output)'''
        '''game = GameState()
        count = 0
        while True:
            action = self.getAction()
            survived = game.step(action)
            if survived:
                print('timeStep: ', count, ' tree: ', game.getNextTree(),
                      'tree[0] position: ', game.tree._upperTrees[0]['x'])
                count = count + 1
            else:
                print('timeStep: ', count, 'died !', 'tree[0] position: ',
                      game.tree._upperTrees[0]['x'])
                count = 0'''


bird = QLearning()
bird.play()