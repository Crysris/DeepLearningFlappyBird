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

    def predict(self, index):
        '''对某只鸟的状态进行计算出最佳动作 0->不飞  1->飞'''
        layer_1 = tf.add(
            tf.matmul(self._X, self._layer_w_b_1[index]['w']),
            self._layer_w_b_1[index]['b'])
        layer_1 = tf.nn.tanh(layer_1)
        layer_output = tf.add(
            tf.matmul(layer_1, self._layer_w_b_output[index]['w']),
            self._layer_w_b_output[index]['b'])
        return layer_output

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

    def mutation(self):
        '''随机选址权值突变'''
        change = [1, 0.5, -0.5, -1]
        for i in range(self.num):
            prob = np.random.uniform()
            if prob > 0.9:
                self._layer_w_b_1[i][
                    'w'] = self._layer_w_b_1[i]['w'] + change[np.random.
                                                              randint(4)]
                self._layer_w_b_1[i][
                    'b'] = self._layer_w_b_1[i]['b'] + change[np.random.
                                                              randint(4)]
                self._layer_w_b_output[i][
                    'w'] = self._layer_w_b_output[i]['w'] + change[np.random.
                                                                   randint(4)]
                self._layer_w_b_output[i][
                    'b'] = self._layer_w_b_output[i]['b'] + change[np.random.
                                                                   randint(4)]

    def getTopBirdIndex(self, k=4):
        '''返回score最大的5只鸟的index'''
        tuples = [(self.game.birdList[i].score, i) for i in range(self.num)]
        return [val[1] for val in sorted(tuples[:k])]

    def start(self):
        sess = tf.InteractiveSession()
        self.initNetwork()
        self.game = GameState(self.num)
        self.game.start()
        step = 1
        Round = 1
        maxScore = 0
        maxTrees = 0
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state('saved_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('network loaded succeccfully!',
                  checkpoint.model_checkpoint_path)
        else:
            print('network loaded failed!')
        # 10只鸟的tensor
        layer_output = [self.predict(i) for i in range(self.num)]

        while True:
            actions = []
            for i in range(self.num):
                data = self.game.getData()
                actions.append(
                    np.argmax(
                        sess.run(
                            layer_output[i], feed_dict={self._X: data[i]})))
            newRound, score, trees = self.game.geneticStep(actions)
            if newRound:
                Round += 1
                last_layer_1_w_b = self._layer_w_b_1
                last_layer_output_w_b = self._layer_w_b_output
                # score最高的5只鸟权值不变
                idxs = self.getTopBirdIndex()
                self._layer_w_b_1 = []
                self._layer_w_b_output = []
                for i in range(4):
                    self._layer_w_b_1.append(last_layer_1_w_b[i])
                    self._layer_w_b_output.append(last_layer_output_w_b[i])
                for i in range(3):
                    p = np.random.randint(len(idxs))
                    q = np.random.randint(len(idxs))
                    self._layer_w_b_1.append(last_layer_1_w_b[p])
                    self._layer_w_b_1.append(last_layer_1_w_b[q])
                    self._layer_w_b_output.append(last_layer_output_w_b[q])
                    self._layer_w_b_output.append(last_layer_output_w_b[p])
                self.mutation()

            maxScore = max(maxScore, score)
            maxTrees = max(maxTrees, trees)
            if maxTrees > 0 and maxTrees % 1000 == 0:
                saver.save(
                    sess, 'saved_networks/genetic_bird', global_step=maxTrees)
            if maxTrees == 10000:
                break
            print('Round:', Round, '/Step:', step, '/score:', score,
                  '/MaxScore:', maxScore, '/trees:', trees, '/MaxTrees: ',
                  maxTrees)
            step += 1


genetic = Genetic()
genetic.start()