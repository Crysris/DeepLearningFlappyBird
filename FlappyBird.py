import pygame as pg
import sys
from pygame.locals import *
import random
import bisect
import numpy as np

FPS = 30
SCREENWIDTH = 320
SCREENHEIGHT = 640
# SCREENWIDTH=1000
# SCREENHIEGHT=640

# 上下2棵树间隙高度
TREEGAPSIZE = 170
# 鸟可飞行空间高度
BASTY = SCREENHEIGHT * 0.863
IMAGES, HITMASKS = {}, {}
BIRDS = []

# 树与树间的距离
DIS = 250
# DIS=250

# 奖励
# 每经过一帧给予1的奖励
FRAME_REWARD = 1
# 每经过一棵树给予50的奖励
PASS_REWARD = 50
# 死亡给予-1000的奖励
DIE_REWARD = -1000
FRAME_PER_SECOND = 1
global SCREEN, FPSCLOCK
pg.init()
FPSCLOCK = pg.time.Clock()
SCREEN = pg.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pg.display.set_caption('Flappy Bird')


class GameState(object):
    def __init__(self, Num=1):
        # 草地
        IMAGES['img_ground'] = pg.image.load('src/ground.png').convert_alpha()
        # 背景图片
        IMAGES['background'] = pg.image.load('src/bg4.png').convert()

        # 上下两棵树
        TREE_LIST = ['src/tree_up.png', 'src/tree_down.png']
        IMAGES['tree'] = (pg.image.load(TREE_LIST[0]).convert_alpha(),
                          pg.image.load(TREE_LIST[1]).convert_alpha())
        # 鸟图片
        for i in range(10):
            BIRDS.append(('src/birds_upflap_' + str(i) + '.png',
                          'src/birds_downflap_' + str(i) + '.png'))
        IMAGES['player'] = []
        for i in range(Num):
            index = np.random.randint(10)
            IMAGES['player'].append(
                (pg.image.load(BIRDS[index][0]).convert_alpha(),
                 pg.image.load(BIRDS[index][1]).convert_alpha()))

        # 草地移动速度
        self.lSpeed = 4
        self.fSpeed = 200
        self.num = Num
        # 求出鸟，草地，树的hitmask
        HITMASKS['player'] = (self.getHitMask(IMAGES['player'][0][0]),
                              self.getHitMask(IMAGES['player'][0][1]))
        HITMASKS['img_ground'] = self.getHitMask(IMAGES['img_ground'])
        HITMASKS['tree'] = (self.getHitMask(IMAGES['tree'][0]),
                            self.getHitMask(IMAGES['tree'][1]))

    def start(self):
        self.birdList = []
        for i in range(self.num):
            bird = Bird(i)
            self.birdList.append(bird)
        self.tree = Tree()
        self.basex = 0
        self._baseShift = IMAGES['img_ground'].get_width() - SCREENWIDTH

    def geneticStep(self, action):
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN
                                      and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()

        for i in range(self.num):
            # 鸟死亡后不再检查
            if not self.birdList[i].survived:
                continue
            # 高度超出一定距离后不再跳跃 action=1->跳跃
            if action[i] > 0.5 and self.birdList[i]._playery > 0:
                self.birdList[i]._initSpeed = self.birdList[i]._flapSpeed
                self.birdList[i]._hasFlapped = True

            playerMidPos = self.birdList[i]._playerx + IMAGES['player'][i][0].get_width(
            ) / 2
            for tree in self.tree._upperTrees:
                treeMidPos = tree['x'] + IMAGES['tree'][0].get_width() / 2
                if treeMidPos <= playerMidPos < treeMidPos - self.tree._treeSpeed:
                    # 每飞过一棵树就给予50的reword
                    # self.birdList[i].score += 50
                    self.birdList[i].passedTrees += 1
                    break
            # 检查是否已产生撞击
            if self.check(i)[0]:
                self.birdList[i].survived = False
            else:
                # 每存活着一帧就给予1的reword
                self.birdList[i].score += 1

        newRound = False
        if [self.birdList[i].survived
                for i in range(self.num)] == [0] * self.num:
            self.start()
            newRound = True
        SCREEN.blit(IMAGES['background'], (0, 0))
        self.tree.start()
        self.groundMoving(self.fSpeed)
        for i in range(self.num):
            if self.birdList[i].survived:
                self.birdList[i].flap()
        pg.display.update()
        FPSCLOCK.tick(FPS)
        # 返回最大reword和飞过树的数目
        return newRound, np.max([
            self.birdList[i].score for i in range(self.num)
        ]), np.max([self.birdList[i].passedTrees for i in range(self.num)])

    def getData(self):
        treeIndex = self.getNextTree()
        data = []
        for i in range(self.num):
            dx = self.tree._lowerTrees[treeIndex]['x'] + IMAGES['tree'][0].get_width(
            ) / 2 - self.birdList[i]._playerx - IMAGES['player'][0][0].get_width(
            ) / 2
            dy = self.tree._lowerTrees[treeIndex]['y'] - TREEGAPSIZE / 2 - self.birdList[i]._playery - IMAGES['player'][0][0].get_height(
            ) / 2
            data.append([[dx, dy]])
        return data

    def groundMoving(self, speed):
        self.basex = -((-self.basex + speed) % self._baseShift)
        SCREEN.blit(IMAGES['img_ground'], (self.basex, BASTY))

    def getHitMask(self, image):
        '''用图片的alpha值检测是否发生碰撞'''
        mask = []
        for i in range(image.get_width()):
            a = []
            for j in range(image.get_height()):
                a.append(bool(image.get_at((i, j))[3]))
            mask.append(a)
        return mask

    def check(self, index):
        '''检查是否坠地或者撞到树'''
        playerHeight = IMAGES['player'][index][0].get_height()
        playerWidth = IMAGES['player'][index][0].get_width()
        # 坠地
        if self.birdList[index]._playery + playerHeight >= BASTY:
            return [True, True]
        else:
            playerRect = pg.Rect(self.birdList[index]._playerx,
                                 self.birdList[index]._playery, playerWidth,
                                 playerHeight)
            treeWidth = IMAGES['tree'][0].get_width()
            treeHeight = IMAGES['tree'][0].get_height()
            for uTree, lTree in zip(self.tree._upperTrees,
                                    self.tree._lowerTrees):
                uTreeRect = pg.Rect(uTree['x'], uTree['y'], treeWidth,
                                    treeHeight)
                lTreeRect = pg.Rect(lTree['x'], lTree['y'], treeWidth,
                                    treeHeight)
                if self.pixelCollision(
                        playerRect, uTreeRect,
                        HITMASKS['player'][self.birdList[index]._dirIndex],
                        HITMASKS['tree'][0]) or self.pixelCollision(
                            playerRect, lTreeRect,
                            HITMASKS['player'][self.birdList[index]._dirIndex],
                            HITMASKS['tree'][1]):
                    return [True, False]
            return [False, False]

    def getNextTree(self):
        '''返回鸟下一个要飞过的树的index'''
        treeX = [
            x['x'] + IMAGES['tree'][0].get_width()
            for x in self.tree._lowerTrees
        ]
        return bisect.bisect_left(
            treeX, SCREENWIDTH * 0.1 + IMAGES['player'][0][0].get_width())

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        '''检查两个物体是否相撞'''
        # 重合部分
        rect = rect1.clip(rect2)
        if rect.width == 0 and rect.height == 0:
            return False
        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y
        for i in range(rect.width):
            for j in range(rect.height):
                if hitmask1[x1 + i][y1 + j] and hitmask2[x2 + i][y2 + j]:
                    return True
        return False


class Bird(object):
    def __init__(self, index=0):
        self._x0 = int(SCREENWIDTH * 0.1)
        self._y0 = int(
            (SCREENHEIGHT - IMAGES['player'][index][0].get_height()) / 2)
        self._playerx = self._x0
        self._playery = self._y0
        self._playerShmVals = {'val': 0, 'dir': 1}
        self._loopIter = 0
        self._dirIndex = 0
        # 初始速度
        self._initSpeed = -9
        # 最大下落速度
        self._maxSpeed = 30
        # 加速
        self._accSpeed = 4
        # 最大跳跃速度
        self._flapSpeed = -25
        self._hasFlapped = False
        self._birdHeight = IMAGES['player'][index][self._dirIndex].get_height()
        self.score = 0
        self.survived = True
        self._index = index
        self.passedTrees = 0

    def flap(self):
        if (self._loopIter + 1) % 3 == 0:
            self._dirIndex ^= 1
        self._loopIter = (self._loopIter + 1) % 30

        # 鸟正常下落
        if self._initSpeed < self._maxSpeed and not self._hasFlapped:
            self._initSpeed += self._accSpeed
        # 跳跃
        if self._hasFlapped:
            self._hasFlapped = False

        self._playery += min(self._initSpeed,
                             BASTY - self._playery - self._birdHeight)
        self._playery = max(self._playery, 0)
        SCREEN.blit(IMAGES['player'][self._index][self._dirIndex],
                    (self._playerx, self._playery))

    def playerShm(self):
        '''鸟上下震荡'''
        if abs(self._playerShmVals['val']) == 8:
            self._playerShmVals['dir'] *= -1
        if self._playerShmVals['dir'] == 1:
            self._playerShmVals['val'] += 1
        else:
            self._playerShmVals['val'] -= 1


class Tree(object):
    def __init__(self):
        # 树移动速度
        self._treeSpeed = -12.5
        newTrees = [self.getRandomTree() for i in range(4)]
        self._upperTrees = []
        self._lowerTrees = []
        self._count = 0
        for i in range(1):
            self._upperTrees.append({
                'x': SCREENWIDTH - DIS * (i) + 150,
                'y': newTrees[i][0]['y']
            })
            self._lowerTrees.append({
                'x': SCREENWIDTH - DIS * (i) + 150,
                'y': newTrees[i][1]['y']
            })

    def start(self):
        # 移动树
        for uTree, lTree in zip(self._upperTrees, self._lowerTrees):
            uTree['x'] += self._treeSpeed
            lTree['x'] += self._treeSpeed
        # 产生新的树
        self._count = (self._count - self._treeSpeed) % DIS
        if self._count == 0:
            newTree = self.getRandomTree()
            self._upperTrees.append(newTree[0])
            self._lowerTrees.append(newTree[1])

        # 移除最前面一棵树
        if self._upperTrees[0]['x'] < -IMAGES['tree'][0].get_width():
            self._upperTrees.pop(0)
            self._lowerTrees.pop(0)

        # 显示出每一棵树
        for uTree, lTree in zip(self._upperTrees, self._lowerTrees):
            SCREEN.blit(IMAGES['tree'][0], (uTree['x'], uTree['y']))
            SCREEN.blit(IMAGES['tree'][1], (lTree['x'], lTree['y']))

    def getRandomTree(self):
        gapY = random.randrange(0, int(BASTY * 0.6 - TREEGAPSIZE))
        gapY += int(BASTY * 0.2)
        treeHeight = IMAGES['tree'][0].get_height()
        treeX = SCREENWIDTH
        return [{
            'x': treeX + 150,
            'y': gapY - treeHeight
        }, {
            'x': treeX + 150,
            'y': gapY + TREEGAPSIZE
        }]
